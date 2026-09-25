# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for the DuoMamba ACCV 2026 release.
_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_DuoMamba',
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        embed_dim=64,
        depths=(2, 5, 20, 6),
        num_heads=(4, 8, 16, 16),
        expand=2,
        chunk_size=64,
        mixer_types=['DuoScanMixer', 'DuoScanMixer', ['DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA'], 'MHSA'],
        d_state=16,
        ngroups=2,
        drop_path_rate=0.25,
        mlp_ratio=4.,
        gating=True,
        lpu=True,
        stem_ver=1,
        downsampler_ver=1,
    ),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150, act_cfg=dict(type='ReLU', inplace=False)),
    auxiliary_head=dict(in_channels=256, num_classes=150, act_cfg=dict(type='ReLU', inplace=False)),
)

# 70.05 M 1048 GFLOPs
#
