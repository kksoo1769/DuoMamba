# COCO object detection and instance segmentation

Complete [installation](../docs/INSTALL.md), including detection dependencies. Commands below start from the repository root.

## Data and weights

```text
/path/to/coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

Training uses the matching [ImageNet-1K classification checkpoint](https://huggingface.co/PangS00oo/DuoMamba/tree/main/classification) from Hugging Face. Download `duomamba_tiny.pth`, `duomamba_small.pth`, or `duomamba_base.pth` for the chosen backbone. Each file stores weights under `model`. Evaluation uses a complete Mask R-CNN checkpoint from the [result tables](../README.md#results). Download the checkpoint matching the variant and schedule; each file stores the complete task model under `state_dict`. Original experiment logs are linked alongside the weights.

## Training

```bash
cd detection
bash tools/dist_train.sh configs/duomamba/mask_rcnn_duomamba_fpn_coco_tiny.py 8 \
    --data-path /path/to/coco \
    --work-dir ./work_dirs/duomamba_tiny_1x \
    --cfg-options model.backbone.pretrained=/path/to/duomamba_tiny.pth
```

| Variant | 1× / 12 epochs | 3× / 36 epochs + multi-scale training |
|---|---|---|
| Tiny | `mask_rcnn_duomamba_fpn_coco_tiny.py` | `mask_rcnn_duomamba_fpn_coco_tiny_3x.py` |
| Small | `mask_rcnn_duomamba_fpn_coco_small.py` | `mask_rcnn_duomamba_fpn_coco_small_3x.py` |
| Base | `mask_rcnn_duomamba_fpn_coco_base.py` | Not supplied |

All paths above are relative to `configs/duomamba/`. The recipe uses 2 images per GPU on 8 GPUs (total 16). Changing GPU count changes the total batch unless you also adjust the configuration. The 1× schedule has 500 warmup iterations; the 3× schedule has 1000.

## Evaluation

```bash
# From detection/
bash tools/dist_test.sh configs/duomamba/mask_rcnn_duomamba_fpn_coco_tiny.py \
    /path/to/mask_rcnn_duomamba_tiny_coco_1x.pth 8 \
    --data-path /path/to/coco
```

The default `pretrained=None` avoids trying to load an extra classification checkpoint when evaluating a complete detector. Always pass `model.backbone.pretrained` for the paper's training recipe; without it, the backbone starts from random initialization. Invalid explicit checkpoint paths raise an error.

`--data-path` sets training, validation, and test roots and COCO annotation paths. Additional `--cfg-options` are preserved and take precedence. Set `PORT=29501` before the command when another distributed job uses the default port.
