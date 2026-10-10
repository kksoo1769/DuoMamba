<div align="center">

# DuoMamba
### A Vision Backbone with Duo Scan and Resolution-Aware Mixer Scheduling

**Accepted to ACCV 2026**

Gwangsoo Kim,
Seungkyu Oh,
Dohhoon Kim,
Nirmal Adhikari,
Bimal Thapa Magar,
Suan Lee,
Wookey Lee

[Installation](docs/INSTALL.md) · [Classification](classification/README.md) · [Detection](detection/README.md) · [Segmentation](segmentation/README.md) · [Citation](#citation)

</div>

Official implementation of **DuoMamba**, a hierarchical vision backbone combining state space models (SSMs) and self-attention.

- **Duo Scan** splits the heads into row and column groups and scans each group in both directions, covering four spatial directions at twice the unidirectional scan cost.
- **Resolution-aware mixer scheduling** is a fixed placement heuristic: Mamba2 mixers in the first two stages, interleaved with self-attention in stage 3, followed by self-attention in stage 4. The schedule is chosen at the pre-training resolution and stays fixed at other input resolutions; it is not a guarantee of an optimal placement.

## Release status

- **September 21, 2026:** accepted to ACCV 2026.
- The camera-ready manuscript is in preparation. There is no public arXiv or proceedings link yet.
- Training and evaluation code and Tiny/Small/Base configurations are included. ImageNet-1K pretrained weights, COCO Mask R-CNN and ADE20K UPerNet checkpoints, and original experiment logs are hosted on [Hugging Face](https://huggingface.co/PangS00oo/DuoMamba).
- Reported results below are from the ACCV manuscript. They have not been remeasured during this repository cleanup.

## Getting started

```bash
git clone https://github.com/kksoo1769/DuoMamba.git
cd DuoMamba
```

The model requires Linux, an NVIDIA GPU, and CUDA extensions. Follow [installation](docs/INSTALL.md) first, then the relevant task guide. Each task guide starts from the repository root.

```text
classification/   ImageNet-1K training, evaluation, and shared backbone
  models/         Duo Scan, self-attention, and DuoMamba
  configs/duomamba/  Tiny / Small / Base configurations
detection/        Mask R-CNN on COCO 2017
segmentation/     UPerNet on ADE20K
docs/             Installation and release notes
requirements/     Task-specific and validation dependencies
scripts/          Configuration validation and CUDA smoke check
LICENSES/         Third-party license texts
```

The implementation lives in `classification/models/duomamba.py`: `DuoMamba` is the backbone, `DuoMambaBlock` is its residual block, and `DuoScanMixer` implements Duo Scan. Classification configs use `MODEL.DUOMAMBA`, and all three tasks use `DuoScanMixer` in their mixer schedules. See [release notes](docs/RELEASE_NOTES.md) for the included recipes and verification scope.

## Pretrained weights

The result tables below link to the weights and original experiment logs on [Hugging Face](https://huggingface.co/PangS00oo/DuoMamba). ImageNet-1K checkpoints contain `model` weights for classification evaluation or backbone initialization. Mask R-CNN and UPerNet checkpoints contain the complete task model under `state_dict`. These evaluation checkpoints omit optimizer and scheduler states. See the task guides and [artifact manifest](docs/ARTIFACTS.json) for configurations and files.

Hugging Face groups weights and logs by task: `classification/`, `detection/`, and `segmentation/`. Each task directory contains its checkpoints and a `logs/` subdirectory.

## Results

### ImageNet-1K classification

224 × 224 input, 300 epochs, trained from scratch.

| Model | #Params | FLOPs | Top-1 Acc (%) | Weights | Logs |
|-------|---------|-------|---------------|---------|------|
| DuoMamba-T | 28M  | 4.8G  | 84.0 | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/classification/duomamba_tiny.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/classification/logs/tiny/log_rank0.txt) |
| DuoMamba-S | 41M  | 7.2G  | 84.7 | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/classification/duomamba_small.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/classification/logs/small/log_rank0.txt) |
| DuoMamba-B | 91M  | 15.4G | 85.4 | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/classification/duomamba_base.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/classification/logs/base/log_rank0.txt) |

### COCO object detection and instance segmentation

Mask R-CNN with ImageNet-1K initialization. FLOPs are reported at 1280 × 800. AP<sup>b</sup> denotes bounding-box AP; AP<sup>m</sup> denotes mask AP.

**1× schedule (12 epochs)**

| Backbone | AP<sup>b</sup> | AP<sup>b</sup><sub>50</sub> | AP<sup>b</sup><sub>75</sub> | AP<sup>m</sup> | AP<sup>m</sup><sub>50</sub> | AP<sup>m</sup><sub>75</sub> | #Params | FLOPs | Weights | Logs |
|----------|------|---------|---------|------|---------|---------|---------|-------|---------|------|
| DuoMamba-T | 48.3 | 70.2 | 53.3 | 43.6 | 67.5 | 47.2 | 48M | 301G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/detection/mask_rcnn_duomamba_tiny_coco_1x.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/detection/logs/mask_rcnn_duomamba_fpn_coco_tiny/20260209_143659.log) |
| DuoMamba-S | 49.7 | 71.7 | 54.6 | 44.5 | 68.7 | 47.7 | 61M | 369G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/detection/mask_rcnn_duomamba_small_coco_1x.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/detection/logs/mask_rcnn_duomamba_fpn_coco_small/20260217_162126.log) |
| DuoMamba-B | 50.6 | 72.5 | 55.6 | 45.2 | 69.6 | 48.8 | 111M | 564G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/detection/mask_rcnn_duomamba_base_coco_1x.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/detection/logs/mask_rcnn_duomamba_fpn_coco_base/20260213_150303.log) |

**3× schedule (36 epochs) with multi-scale training**

| Backbone | AP<sup>b</sup> | AP<sup>b</sup><sub>50</sub> | AP<sup>b</sup><sub>75</sub> | AP<sup>m</sup> | AP<sup>m</sup><sub>50</sub> | AP<sup>m</sup><sub>75</sub> | #Params | FLOPs | Weights | Logs |
|----------|------|---------|---------|------|---------|---------|---------|-------|---------|------|
| DuoMamba-T | 49.9 | 71.3 | 54.7 | 44.3 | 68.5 | 47.8 | 48M | 301G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/detection/mask_rcnn_duomamba_tiny_coco_3x.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/detection/logs/mask_rcnn_duomamba_fpn_coco_tiny_3x/20260210_145436.log) |
| DuoMamba-S | 51.1 | 72.3 | 56.0 | 45.3 | 69.5 | 49.1 | 61M | 369G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/detection/mask_rcnn_duomamba_small_coco_3x.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/detection/logs/mask_rcnn_duomamba_fpn_coco_small_3x/20260219_003230.log) |

### ADE20K semantic segmentation

UPerNet, 512 × 512 training crops, 160k iterations, total batch size 16. SS/MS denote single-scale/multi-scale evaluation; reported FLOPs follow the manuscript's 2048 × 512 convention.

| Backbone | mIoU (SS) | mIoU (MS) | #Params | FLOPs | Weights | Logs |
|----------|-----------|-----------|---------|-------|---------|------|
| DuoMamba-T | 49.5 | 50.0 | 57M | 978G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/segmentation/upernet_duomamba_tiny_ade20k.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/segmentation/logs/upernet_duomamba_8xb2-160k_ade20k-512x512_tiny/20260212_122903.log) |
| DuoMamba-S | 50.3 | 51.3 | 70M | 1048G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/segmentation/upernet_duomamba_small_ade20k.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/segmentation/logs/upernet_duomamba_8xb2-160k_ade20k-512x512_small/20260219_000549.log) |
| DuoMamba-B | 52.1 | 52.7 | 122M | 1250G | [Weights](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/segmentation/upernet_duomamba_base_ade20k.pth) | [Logs](https://huggingface.co/PangS00oo/DuoMamba/blob/main/segmentation/logs/upernet_duomamba_8xb2-160k_ade20k-512x512_base/20260215_075214.log) |

## Citation

The entry below is provisional while the camera-ready version is being prepared. Proceedings pages and DOI will be added when available.

```bibtex
@inproceedings{kim2026duomamba,
  title     = {DuoMamba: A Vision Backbone with Duo Scan and Resolution-Aware Mixer Scheduling},
  author    = {Kim, Gwangsoo and Oh, Seungkyu and Kim, Dohhoon and Adhikari, Nirmal and Thapa Magar, Bimal and Lee, Suan and Lee, Wookey},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026},
  note      = {Accepted; camera-ready version in preparation}
}
```

Machine-readable author and citation metadata are provided in [CITATION.cff](CITATION.cff).

## Acknowledgements and licensing

This implementation builds on [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [MLLA](https://github.com/LeapLabTHU/MLLA), [Mamba](https://github.com/state-spaces/mamba), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

A project-wide license for the original DuoMamba contributions has not yet been selected. Third-party portions retain their existing licenses and copyright notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The third-party license files do not grant a blanket license to the whole repository.
