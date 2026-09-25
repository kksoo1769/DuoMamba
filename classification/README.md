# ImageNet-1K classification

Complete [installation](../docs/INSTALL.md). Commands below start from the repository root.

## Data

The existing loader expects these directory names, with both splits organized by ImageNet class ID:

```text
/path/to/imagenet/
  ILSVRC2012_img_train/
    n01440764/*.JPEG
    ...
  ILSVRC2012_img_val/
    n01440764/*.JPEG
    ...
```

If your directories are named `train` and `val`, create corresponding `ILSVRC2012_img_train` and `ILSVRC2012_img_val` symlinks. Validation images must be arranged into class directories. The current loader constructs **both splits even during evaluation**.

## Training

```bash
cd classification
torchrun --standalone --nproc_per_node=8 main.py \
    --cfg configs/duomamba/tiny.yaml \
    --data-path /path/to/imagenet \
    --output ./output --tag tiny_300ep
```

Select `small.yaml` or `base.yaml` for the larger models. These public configs use the ACCV supplementary recipe:

| Variant | Batch per GPU (8 GPUs) | Effective batch | Learning rate | MESA coefficient |
|---|---:|---:|---:|---:|
| Tiny | 128 | 1024 | 0.001 | 1.0 |
| Small | 256 | 2048 | 0.002 | 1.5 |
| Base | 256 | 2048 | 0.002 | 2.0 |

All variants use 300 epochs, 20 warmup epochs, bfloat16 AMP, and EMA decay 0.9999. MESA distillation starts at epoch 75. The script scales learning rate by effective batch size relative to 512. Effective batch = batch per GPU × GPU count × accumulation steps.

For less GPU memory, reduce `--batch-size` and increase `--accumulation-steps` to preserve the effective batch. For example, Tiny on 8 GPUs with `--batch-size 64 --accumulation-steps 2` keeps an effective batch of 1024.

## Evaluation

Checkpoints must be obtained separately. No pretrained weights are bundled in this release.

```bash
# From classification/
torchrun --standalone --nproc_per_node=1 main.py \
    --cfg configs/duomamba/tiny.yaml \
    --data-path /path/to/imagenet \
    --eval --pretrained /path/to/duomamba_tiny.pth \
    --output ./output --tag tiny_eval \
    --opts TRAIN.AUTO_RESUME False
```

A classification checkpoint contains `model` and optionally `model_ema` state dictionaries. The evaluator logs both when available. Match the config to the checkpoint variant. `--eval` requires a checkpoint; evaluation without weights is rejected.

## Resume

```bash
# From classification/
torchrun --standalone --nproc_per_node=8 main.py \
    --cfg configs/duomamba/tiny.yaml \
    --data-path /path/to/imagenet \
    --resume /path/to/latest_ckpt.pth \
    --output ./output --tag tiny_300ep
```

Output is stored under `<output>/<model-name>/<tag>`. A matching output directory can auto-resume; use a new tag for an independent run.

Model settings use `MODEL.DUOMAMBA`, for example `--opts MODEL.DUOMAMBA.DROP_PATH_RATE 0.2`. The SSM mixer is named `DuoScanMixer`; attention remains `MHSA`. The release includes Tiny, Small, and Base presets under `configs/duomamba/`. CLI overrides retain their usual order, with the last value winning.
