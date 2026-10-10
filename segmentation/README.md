# ADE20K semantic segmentation

Complete [installation](../docs/INSTALL.md), including segmentation dependencies. Commands below start from the repository root.

## Data and weights

```text
/path/to/ADEChallengeData2016/
  images/
    training/
    validation/
  annotations/
    training/
    validation/
```

Training uses the matching [ImageNet-1K classification checkpoint](https://huggingface.co/PangS00oo/DuoMamba/tree/main/classification) from Hugging Face. Download `duomamba_tiny.pth`, `duomamba_small.pth`, or `duomamba_base.pth` for the chosen backbone. Each file stores weights under `model`. Evaluation uses a complete UPerNet checkpoint from the [result tables](../README.md#results). Download the checkpoint matching the variant and schedule; each file stores the complete task model under `state_dict`. Original experiment logs are linked alongside the weights. Each Hugging Face experiment folder contains its checkpoint and corresponding log together.

## Training

```bash
cd segmentation
bash tools/dist_train.sh configs/duomamba/upernet_duomamba_8xb2-160k_ade20k-512x512_tiny.py 8 \
    --data-path /path/to/ADEChallengeData2016 \
    --work-dir ./work_dirs/duomamba_tiny \
    --cfg-options model.backbone.pretrained=/path/to/duomamba_tiny.pth
```

Replace `_tiny.py` with `_small.py` or `_base.py` for the other variants. Training uses 512 × 512 crops, 160k iterations, and 2 images per GPU on 8 GPUs (total 16). Changing GPU count changes the total batch unless the configuration is adjusted.

## Single-scale evaluation

```bash
# From segmentation/
bash tools/dist_test.sh configs/duomamba/upernet_duomamba_8xb2-160k_ade20k-512x512_tiny.py \
    /path/to/upernet_duomamba_tiny_ade20k.pth 8 \
    --data-path /path/to/ADEChallengeData2016
```

## Multi-scale evaluation

```bash
# From segmentation/
bash tools/dist_test.sh configs/duomamba/upernet_duomamba_8xb2-160k_ade20k-512x512_tiny.py \
    /path/to/upernet_duomamba_tiny_ade20k.pth 8 \
    --data-path /path/to/ADEChallengeData2016 --tta
```

Test-time augmentation (TTA) combines scales 0.75, 1.0, and 1.25 with horizontal flips, as in the ACCV supplementary material.

The default `pretrained=None` avoids requiring classification weights when evaluating a complete segmentor. Pass `model.backbone.pretrained` for the paper's training recipe. Invalid explicit paths raise an error. Additional `--cfg-options` are preserved and take precedence over `--data-path`.
