# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for the DuoMamba ACCV 2026 release.
_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
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
# box: 0.483 0.702 0.533 0.325 0.521 0.625
# mask: 0.436 0.675 0.472 0.252 0.466 0.622

# ACCV supplementary recipe: 500 warmup iterations for the 1x schedule.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[8, 11], gamma=0.1),
]
