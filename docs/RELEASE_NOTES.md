# DuoMamba ACCV 2026 code release

## Included code

- ImageNet-1K classification with DuoMamba Tiny, Small, and Base configurations.
- COCO 2017 Mask R-CNN detection and instance segmentation configurations.
- ADE20K UPerNet semantic segmentation configurations.
- Training and evaluation launchers, installation instructions, and configuration checks.

Datasets are not bundled. Model-only ImageNet-1K pretrained checkpoints for Tiny, Small, and Base are available on [Hugging Face](https://huggingface.co/PangS00oo/DuoMamba); they contain no training history, optimizer state, or accuracy records. Complete Mask R-CNN and UPerNet evaluation checkpoints and original experiment logs are also available. The result tables link to each artifact; [ARTIFACTS.json](ARTIFACTS.json) records source-file mappings, sizes, and SHA-256 checksums. Reported results in the README come from the ACCV 2026 manuscript.

## Model and training settings

`classification/models/duomamba.py` defines the shared backbone, its blocks, and Duo Scan. The classification configurations use `MODEL.DUOMAMBA`; detection and segmentation register the backbone as `MM_DuoMamba`.

Classification uses 300 epochs and bfloat16 precision. The per-GPU batch sizes are 128, 256, and 256 for Tiny, Small, and Base on eight GPUs. The Mask R-CNN 1× schedule uses 500 warmup iterations; the 3× schedule uses 1000. ADE20K training uses 512 × 512 crops, 160k iterations, and a total batch size of 16.

These configurations specify the ACCV recipe. The published model-only checkpoints intentionally omit training records; the original experiment logs are now linked in the README. Loading a configuration successfully does not establish accuracy reproduction.

## Execution and verification

The task adapters register DuoMamba separately for MMDetection and MMSegmentation. Their launchers preserve explicit config overrides when `--data-path` is used. Detection joins COCO annotation paths without requiring a trailing slash. Classification evaluation requires weights, and explicit invalid backbone checkpoint paths raise an error.

The configuration checker parses Python sources, resolves the included OpenMMLab configurations, and checks manuscript settings, configuration inheritance, CLI precedence, chunk-size type changes, and model-factory wiring without CUDA. A separate CUDA smoke check runs a forward and backward pass with random input. The published checkpoint tensors were compared with the original training checkpoints, and their `model` keys were verified with safe CPU loading. This release preparation did not independently measure accuracy or verify CUDA inference; installation instructions describe a target, not a certified reproduction environment.

## Licensing

A project-wide license for original DuoMamba contributions has not been selected. Existing third-party notices and license texts apply to their respective portions; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## October 10, 2026 artifact release

Added five COCO Mask R-CNN checkpoints, three ADE20K UPerNet checkpoints, and eleven original experiment logs to Hugging Face. The dense prediction checkpoints preserve every `state_dict` tensor and basic epoch/iteration and dataset metadata, while removing optimizer, scheduler, message-hub state, and the embedded training configuration. CPU checks confirmed tensor equality to the source checkpoints, finite tensors, and loading with `weights_only=True`. Log copies were verified with SHA-256. This artifact release did not run dataset evaluation or CUDA inference.

Weights and logs on Hugging Face are grouped by task under `classification/`, `detection/`, and `segmentation/`; each task contains its own `logs/` directory. The artifact manifest and checksums cover all eleven checkpoints and eleven logs.
