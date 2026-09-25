# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted for DuoMamba; see LICENSES/Mamba-Apache-2.0.txt.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_ref

try:
    from utils.utils import is_flopcounting
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import is_flopcounting

try:
    from mamba_util import scaled_dot_product_attention_ref
except ImportError:
    from .mamba_util import scaled_dot_product_attention_ref


class MHSA(nn.Module):
    """Multi-head self-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        head_dim=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        norm_type='ln',
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.norm_type = norm_type
        self.softmax_scale = softmax_scale
        self.causal = causal

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.num_heads == self.num_heads_kv, "Only support num_heads == num_heads_kv"
        if head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = head_dim if head_dim is not None else self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        self.in_proj = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(out_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def forward(self, x, H=None, W=None):
        qkv = self.in_proj(x)

        q, kv = qkv.split([self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1)
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        k, v = kv.unbind(dim=-3)

        if is_flopcounting():
            context = scaled_dot_product_attention_ref(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=self.causal, scale=self.softmax_scale,
            ).transpose(1, 2)
        else:
            context = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=self.causal, scale=self.softmax_scale,
            ).transpose(1, 2)

        context = rearrange(context, "... h d -> ... (h d)")
        out = self.out_proj(context)
        return out
