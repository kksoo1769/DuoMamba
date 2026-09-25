# Duo Scan and DuoMamba implementation, adapted from Mamba2.
# Upstream: Tri Dao and Albert Gu; see LICENSES/Mamba-Apache-2.0.txt.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, rms_norm_ref
from mamba_ssm.ops.triton.layernorm_gated import LayerNorm as LayerNormGated
from einops import rearrange

import logging
import math
import copy
try:
    from mamba_util import PatchMerging, Stem, Mlp, StemV2, StemV0, PatchMergingV2, PatchMergingV0, layer_norm_ref, ResDWC
    from mhsa import MHSA
except ImportError:
    from .mamba_util import PatchMerging, Stem, Mlp, StemV2, StemV0, PatchMergingV2, PatchMergingV0, layer_norm_ref, ResDWC
    from .mhsa import MHSA

from fvcore.nn import flop_count, parameter_count
try:
    from utils.utils import is_flopcounting
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import is_flopcounting


class DuoScanMixer(nn.Module):
    """Duo Scan mixer with axis-factorized bidirectional scanning."""
    def __init__(
        self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=1,
        headdim=64, #default to 64
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, #default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        d_state=64,
        gating=True,
        norm_type='ln',
        norm_before_gate=False,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        assert self.nheads % 2 == 0
        assert ngroups % 2 == 0
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.gating = gating
        self.norm_type = norm_type
        self.norm_before_gate = norm_before_gate
        if self.gating:
            # Order: [z, x, B, C, dt]
            d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        else:
            # Order: [x, B, C, dt]
            d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            groups=self.conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv2d.weight, -self.conv_init, self.conv_init)
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.norm_type == 'ln':
            self.norm = LayerNormGated(self.d_inner, norm_before_gate=norm_before_gate, **factory_kwargs) if self.gating else nn.LayerNorm(self.d_inner, **factory_kwargs)
        elif self.norm_type == 'rmsn':
            self.norm = RMSNormGated(self.d_inner, norm_before_gate=norm_before_gate, **factory_kwargs) if self.gating else RMSNorm(self.d_inner, **factory_kwargs)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, H, W):
        """
        u: (B,L,D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.gating:
            z, xBC, dt = zxbcdt.split([self.d_inner, self.conv_dim, self.nheads], dim=-1)
        else:
            xBC, dt = zxbcdt.split([self.conv_dim, self.nheads], dim=-1)

        #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

        # Split into 3 main branches: X, B, C
        x, B, C = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)

        # Duo Scan: row/column head split with bidirectional scanning
        x_row, x_col = x.chunk(2, dim=-2)  # (b l h/2 p)
        dt_row, dt_col = dt.chunk(2, dim=-1)  # (b l h/2)
        B_row, B_col = B.chunk(2, dim=-2)  # (b l g/2 n)
        C_row, C_col = C.chunk(2, dim=-2)  # (b l g/2 n)
        A_row, A_col = A.chunk(2, dim=0)  # (h/2)
        D_row, D_col = self.D.chunk(2, dim=0)  # (h/2)

        y_row = self.bi_scan(
            x_row, dt_row, B_row, C_row, A_row, D_row, dt_limit_kwargs, self.chunk_size,
        )

        x_col = rearrange(x_col, "b (H W) ... -> b (W H) ...", H=H, W=W)
        dt_col = rearrange(dt_col, "b (H W) ... -> b (W H) ...", H=H, W=W)
        B_col = rearrange(B_col, "b (H W) ... -> b (W H) ...", H=H, W=W)
        C_col = rearrange(C_col, "b (H W) ... -> b (W H) ...", H=H, W=W)

        y_col = self.bi_scan(
            x_col, dt_col, B_col, C_col, A_col, D_col, dt_limit_kwargs, self.chunk_size,
        )
        y_col = rearrange(y_col, "b (W H) ... -> b (H W) ...", H=H, W=W)

        y = torch.cat([y_row, y_col], dim=-2)  # (b l h p)

        y = rearrange(y, 'b l h p -> b l (h p)')


        if self.gating:
            if not is_flopcounting():
                y = self.norm(y, z)
            else:
                if self.norm_type == 'ln':
                    y = layer_norm_ref(y, self.norm.weight, self.norm.bias, z, norm_before_gate=self.norm_before_gate)
                else:
                    y = rms_norm_ref(y, self.norm.weight, self.norm.bias, z, norm_before_gate=self.norm_before_gate)
        else:
            if is_flopcounting() and isinstance(self.norm, RMSNorm):
                y = rms_norm_ref(y, self.norm.weight, self.norm.bias)
            else:
                y = self.norm(y)

        out = self.out_proj(y)

        return out

    def bi_scan(self, x_in, dt_in, B_in, C_in, A_h, D_h, dt_limit_kwargs, chunk_size):
        if not is_flopcounting():
            y_fwd = mamba_chunk_scan_combined(
                x_in, dt_in, A_h, B_in, C_in,
                chunk_size=chunk_size,
                D=D_h,
                **dt_limit_kwargs,
            )
        else:
            x_pad, dt_pad, B_pad, C_pad, L_orig = self.pad_all_for_ref(
                x_in, dt_in, B_in, C_in, chunk_size
            )
            y_fwd = ssd_chunk_scan_combined_ref(
                x_pad, dt_pad, A_h, B_pad, C_pad,
                chunk_size=chunk_size, D=D_h
            )[:, :L_orig]

        x_rev = x_in.flip(1)
        dt_rev = dt_in.flip(1)
        B_rev = B_in.flip(1)
        C_rev = C_in.flip(1)

        if not is_flopcounting():
            y_bwd = mamba_chunk_scan_combined(
                x_rev, dt_rev, A_h, B_rev, C_rev,
                chunk_size=chunk_size,
                D=D_h,
                **dt_limit_kwargs,
            )
        else:
            x_pad, dt_pad, B_pad, C_pad, L_orig = self.pad_all_for_ref(
                x_rev, dt_rev, B_rev, C_rev, chunk_size
            )
            y_bwd = ssd_chunk_scan_combined_ref(
                x_pad, dt_pad, A_h, B_pad, C_pad,
                chunk_size=chunk_size, D=D_h
            )[:, :L_orig]

        y = y_fwd + y_bwd.flip(1)
        return y

    def pad_all_for_ref(self, x, dt, B, C, chunk_size):
        # x: (B, L, H, P), dt: (B, L, H), B/C: (B, L, G, N)
        L_orig = x.size(1)
        pad_len = (-L_orig) % chunk_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        return x, dt, B, C, L_orig



class DuoMambaBlock(nn.Module):
    r""" DuoMamba Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int | list[int] | tuple[int]): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, expand=2, ngroups=1, chunk_size=256, d_state = 64,
                 layer_idx=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_cpe = kwargs.get('lpu')

        self.cpe1 = ResDWC(dim, kernel_size=3) if self.use_cpe else None
        self.norm1 = norm_layer(dim)
        mixer_type = kwargs.get("mixer_type")
        if mixer_type == 'DuoScanMixer':
            self.mixer = DuoScanMixer(d_model=dim, d_state=d_state, d_conv=3, expand=expand,
                               headdim=dim*expand // num_heads, ngroups=ngroups,
                               chunk_size=chunk_size, layer_idx=layer_idx, **kwargs)
        elif mixer_type == 'MHSA':
            self.mixer = MHSA(
                embed_dim=dim, num_heads=num_heads, head_dim=dim // num_heads,
                norm_type=kwargs.get('norm_type', 'ln'), qkv_proj_bias=qkv_bias)
        else:
            raise ValueError(f"Unsupported mixer_type: {mixer_type}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = ResDWC(dim, kernel_size=3) if self.use_cpe else None
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        if (H is None) or (W is None):
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"

        if self.use_cpe:
            x = self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        if is_flopcounting() and isinstance(self.norm1, RMSNorm):
            x = rms_norm_ref(x, self.norm1.weight, self.norm1.bias)
        else:
            x = self.norm1(x)

        # Mixer
        x = self.mixer(x, H, W)
        x = shortcut + self.drop_path(x)
        if self.use_cpe:
            x = self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        # FFN
        if is_flopcounting() and isinstance(self.norm2, RMSNorm):
            x = rms_norm_ref(x, self.norm2.weight, self.norm2.bias)
        else:
            x = self.norm2(x)
        x = shortcut + self.drop_path(self.mlp(x, H, W))
        return x


class BasicLayer(nn.Module):
    """ A basic DuoMamba layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 expand=2, ngroups=1, chunk_size=256, d_state=64, layer_idx=0, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        mixer_type = kwargs.get("mixer_type")
        valid_mixer_types = ("DuoScanMixer", "MHSA")
        if isinstance(mixer_type, (list, tuple)):
            if len(mixer_type) != depth:
                raise ValueError(
                    f"mixer_type length ({len(mixer_type)}) must match depth ({depth})."
                )
            block_mixer_types = list(mixer_type)
            if any(t not in valid_mixer_types for t in block_mixer_types):
                raise ValueError(
                    f"mixer_types elements must be one of: {valid_mixer_types}. "
                    f"Got block mixer types: {block_mixer_types}"
                )
        else:
            if mixer_type not in valid_mixer_types:
                raise ValueError(
                    f"mixer_types elements must be one of: {valid_mixer_types}. "
                    f"Got: {mixer_type}"
                )
            block_mixer_types = [mixer_type] * depth

        if isinstance(num_heads, (list, tuple)):
            if len(num_heads) != depth:
                raise ValueError(
                    f"num_heads length ({len(num_heads)}) must match depth ({depth})."
                )
            block_num_heads = list(num_heads)
        else:
            block_num_heads = [num_heads] * depth

        # build blocks
        self.blocks = nn.ModuleList()
        duoscan_layer_idx = int(layer_idx)
        for i in range(depth):
            block_kwargs = dict(kwargs)
            block_kwargs["mixer_type"] = block_mixer_types[i]
            block_layer_idx = None
            if block_mixer_types[i] == "DuoScanMixer":
                block_layer_idx = duoscan_layer_idx
                duoscan_layer_idx += 1

            self.blocks.append(
                DuoMambaBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=block_num_heads[i],
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                    expand=expand, ngroups=ngroups, chunk_size=chunk_size, d_state=d_state,
                    layer_idx=block_layer_idx, **block_kwargs
                )
            )

        # patch merging layer
        if downsample is not None:
            if downsample is PatchMergingV2 or downsample is PatchMergingV0:
                self.downsample = downsample(input_resolution, dim=dim)
            else:
                self.downsample = downsample(dim, 2*dim)
        else:
            self.downsample = None

    def forward(self, x, H=None, W=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class DuoMamba(nn.Module):
    """DuoMamba: hybrid vision backbone combining DuoScanMixer and MHSA blocks."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depths=[2, 4, 12, 4], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 use_checkpoint=False, expand=2, ngroups=2, chunk_size=256, d_state=16, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        norm_layer = RMSNorm if kwargs.get('norm_type') == 'rmsn' else nn.LayerNorm

        mixer_types = kwargs.get('mixer_types', ['DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA'])
        if len(mixer_types) != self.num_layers:
            raise ValueError(
                f"mixer_types length ({len(mixer_types)}) must match num_layers ({self.num_layers})."
            )
        self.mixer_types = list(mixer_types)
        for i, t in enumerate(self.mixer_types):
            if isinstance(t, (list, tuple)):
                if len(t) != depths[i]:
                    raise ValueError(
                        f"mixer_types[{i}] length ({len(t)}) must match depths[{i}] ({depths[i]})."
                    )
                if any(bt not in {"DuoScanMixer", "MHSA"} for bt in t):
                    raise ValueError(
                        f"mixer_types[{i}] elements must be 'DuoScanMixer' or 'MHSA'. Got: {t}"
                    )
            elif t not in {"DuoScanMixer", "MHSA"}:
                raise ValueError(
                    f"mixer_types must be 'DuoScanMixer' or 'MHSA'. Got: {t}"
                )

        stem_ver = kwargs.get('stem_ver', 0)
        if stem_ver == 0:
            self.patch_embed = StemV0(img_size=img_size, patch_size=patch_size, in_chans=in_chans,embed_dim=embed_dim)
        elif stem_ver == 1:
            self.patch_embed = Stem(in_chans=in_chans, embed_dim=embed_dim)
        elif stem_ver == 2:
            self.patch_embed = StemV2(in_chans=in_chans, embed_dim=embed_dim)
        else:
            raise ValueError

        downsampler_ver = kwargs.get('downsampler_ver', 0)
        if downsampler_ver == 0:
            self.downsample = PatchMergingV0
        elif downsampler_ver == 1:
            self.downsample = PatchMerging
        elif downsampler_ver == 2:
            self.downsample = PatchMergingV2
        else:
            raise ValueError

        patches_resolution = [to_2tuple(img_size)[0] // to_2tuple(patch_size)[0], to_2tuple(img_size)[1] // to_2tuple(patch_size)[1]]

        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        duoscan_layer_idx = 0
        self.layers = nn.ModuleList()

        # Support per-stage chunk_size: can be int or list of ints
        if isinstance(chunk_size, (list, tuple)):
            if len(chunk_size) != self.num_layers:
                raise ValueError(
                    f"chunk_size list length ({len(chunk_size)}) must match num_layers ({self.num_layers})."
                )
            chunk_sizes = list(chunk_size)
        else:
            chunk_sizes = [chunk_size] * self.num_layers

        for i_layer in range(self.num_layers):
            stage_depth = depths[i_layer]
            stage_mixer_type = self.mixer_types[i_layer]
            stage_num_heads = num_heads[i_layer]
            stage_chunk_size = chunk_sizes[i_layer]

            # Handle explicit list of mixer types per block
            if isinstance(stage_mixer_type, (list, tuple)):
                # Already validated in __init__, use directly
                if isinstance(stage_num_heads, (list, tuple)):
                    # User provided explicit num_heads per block
                    if len(stage_num_heads) != stage_depth:
                        raise ValueError(
                            f"num_heads[{i_layer}] length ({len(stage_num_heads)}) "
                            f"must match depths[{i_layer}] ({stage_depth})."
                        )
                else:
                    # Auto-compute num_heads: DuoScanMixer uses stage_num_heads, MHSA uses adjusted
                    attn_num_heads = stage_num_heads
                    if expand != 1:
                        if stage_num_heads % expand != 0:
                            raise ValueError(
                                f"num_heads ({stage_num_heads}) must be divisible by expand ({expand}) "
                                "when using attention blocks with explicit mixer list."
                            )
                        attn_num_heads = int(stage_num_heads // expand)
                    stage_num_heads = [
                        stage_num_heads if mt == "DuoScanMixer" else attn_num_heads
                        for mt in stage_mixer_type
                    ]
            layer_kwargs = dict(kwargs)
            layer_kwargs['mixer_type'] = stage_mixer_type
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=stage_depth,
                               num_heads=stage_num_heads,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=self.downsample if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               expand=expand,
                               ngroups=ngroups,
                               chunk_size=stage_chunk_size,
                               d_state=d_state,
                               layer_idx=duoscan_layer_idx,
                               **layer_kwargs)
            self.layers.append(layer)
            if isinstance(stage_mixer_type, (list, tuple)):
                duoscan_layer_idx += sum(1 for t in stage_mixer_type if t == "DuoScanMixer")
            else:
                duoscan_layer_idx += stage_depth if stage_mixer_type == "DuoScanMixer" else 0

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            logging.warning(f"Error in flop_count: {e}, using default value 1e9")
            return 1e9
        del model, input

        return sum(Gflops.values()) * 1e9

    def forward_features(self, x):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        H, W = H//4, W//4 # downsampled by patch_embed

        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, H, W)
            H, W = H//2, W//2 # downsampled by layer

        if is_flopcounting() and isinstance(self.norm, RMSNorm):
            x = rms_norm_ref(x, self.norm.weight, self.norm.bias)
        else:
            x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class BackboneDuoMamba(DuoMamba):
    """DuoMamba backbone for dense prediction (detection/segmentation)."""
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, **kwargs):
        super().__init__(**kwargs)
        norm_layer = nn.LayerNorm

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.layers[i].dim)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.head
        del self.norm
        del self.avgpool
        self.load_pretrained(pretrained, key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if not ckpt:
            return
        # Do not silently start from random weights when a checkpoint is invalid.
        checkpoint = torch.load(ckpt, map_location="cpu")
        state_dict = checkpoint[key]
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        logging.info("Loaded checkpoint %s from key=%r: %s", ckpt, key, incompatible_keys)


    def forward(self, x):

        def layer_forward(l, x, H=None, W=None):
            for blk in l.blocks:
                x = blk(x, H, W)
            if l.downsample is not None:
                y = x.view(x.shape[0], H, W, -1)
                y = l.downsample(y, H, W)
            else:
                y = x
            return x, y

        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
        H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x, H, W)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                B, L, C = out.shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2)
                outs.append(out.contiguous())
            H, W = int((H-1)/2)+1, int((W-1)/2)+1

        if len(self.out_indices) == 0:
            return x

        return outs
