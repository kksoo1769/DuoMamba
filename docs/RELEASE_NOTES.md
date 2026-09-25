# DuoMamba ACCV 2026 code release

## Included code

- ImageNet-1K classification with DuoMamba Tiny, Small, and Base configurations.
- COCO 2017 Mask R-CNN detection and instance segmentation configurations.
- ADE20K UPerNet semantic segmentation configurations.
- Training and evaluation launchers, installation instructions, and configuration checks.

Datasets and pretrained checkpoints are not bundled. Reported results in the README come from the ACCV 2026 manuscript and have not been remeasured for this release.

## Model and training settings

`classification/models/duomamba.py` defines the shared backbone, its blocks, and Duo Scan. The classification configurations use `MODEL.DUOMAMBA`; detection and segmentation register the backbone as `MM_DuoMamba`.

Classification uses 300 epochs and bfloat16 precision. The per-GPU batch sizes are 128, 256, and 256 for Tiny, Small, and Base on eight GPUs. The Mask R-CNN 1× schedule uses 500 warmup iterations; the 3× schedule uses 1000. ADE20K training uses 512 × 512 crops, 160k iterations, and a total batch size of 16.

These configurations specify the ACCV recipe. Run logs and checkpoint metadata are needed to establish which settings produced the reported numbers. Loading a configuration successfully does not establish accuracy reproduction.

## Execution and verification

The task adapters register DuoMamba separately for MMDetection and MMSegmentation. Their launchers preserve explicit config overrides when `--data-path` is used. Detection joins COCO annotation paths without requiring a trailing slash. Classification evaluation requires weights, and explicit invalid backbone checkpoint paths raise an error.

The configuration checker parses Python sources, resolves the included OpenMMLab configurations, and checks manuscript settings, configuration inheritance, CLI precedence, chunk-size type changes, and model-factory wiring without CUDA. A separate CUDA smoke check runs a forward and backward pass with random input. GPU execution, loading trained checkpoints, and accuracy reproduction were not verified during release preparation. Installation instructions describe an installation target, not a certified reproduction environment.

## Licensing

A project-wide license for original DuoMamba contributions has not been selected. Existing third-party notices and license texts apply to their respective portions; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
