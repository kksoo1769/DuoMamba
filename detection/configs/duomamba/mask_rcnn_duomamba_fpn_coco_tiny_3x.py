# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for the DuoMamba ACCV 2026 release.
_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_DuoMamba',
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        embed_dim=64,
        depths=(2, 4, 10, 5),
        num_heads=(4, 8, 16, 16),
        expand=2,
        chunk_size=64,
        mixer_types=['DuoScanMixer', 'DuoScanMixer', ['DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA'], 'MHSA'],
        d_state=16,
        ngroups=2,
        drop_path_rate=0.2,
        mlp_ratio=4.,
        gating=True,
        lpu=True,
        stem_ver=1,
        downsampler_ver=1,
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
)

# 47.906 M, 301 GFLOPs
# box: 0.499 0.713 0.547 0.342 0.535 0.642
# mask: 0.443 0.685 0.478 0.257 0.473 0.627
