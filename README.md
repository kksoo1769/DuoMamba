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
- Training and evaluation code and Tiny/Small/Base configurations are included. The matching ImageNet-1K pretrained weights are hosted on [Hugging Face](https://huggingface.co/PangS00oo/DuoMamba).
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

| Variant | ImageNet-1K checkpoint | Classification config |
|---|---|---|
| Tiny | [duomamba_tiny.pth](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/duomamba_tiny.pth) | [tiny.yaml](classification/configs/duomamba/tiny.yaml) |
| Small | [duomamba_small.pth](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/duomamba_small.pth) | [small.yaml](classification/configs/duomamba/small.yaml) |
| Base | [duomamba_base.pth](https://huggingface.co/PangS00oo/DuoMamba/resolve/main/duomamba_base.pth) | [base.yaml](classification/configs/duomamba/base.yaml) |

Each checkpoint contains only `model` weights. Download the variant matching your config and use it with `--pretrained` for classification or `model.backbone.pretrained` for detection and segmentation. See the [model card](https://huggingface.co/PangS00oo/DuoMamba) and task guides for details. These files do not contain optimizer state for resuming training.

## Results

### ImageNet-1K classification

224 × 224 input, 300 epochs, trained from scratch.

| Model | #Params | FLOPs | Top-1 Acc (%) |
|-------|---------|-------|---------------|
| DuoMamba-T | 28M  | 4.8G  | 84.0 |
| DuoMamba-S | 41M  | 7.2G  | 84.7 |
| DuoMamba-B | 91M  | 15.4G | 85.4 |

### COCO object detection and instance segmentation

Mask R-CNN with ImageNet-1K initialization. FLOPs are reported at 1280 × 800. AP<sup>b</sup> denotes bounding-box AP; AP<sup>m</sup> denotes mask AP.

**1× schedule (12 epochs)**

| Backbone | AP<sup>b</sup> | AP<sup>b</sup><sub>50</sub> | AP<sup>b</sup><sub>75</sub> | AP<sup>m</sup> | AP<sup>m</sup><sub>50</sub> | AP<sup>m</sup><sub>75</sub> | #Params | FLOPs |
|----------|------|---------|---------|------|---------|---------|---------|-------|
| DuoMamba-T | 48.3 | 70.2 | 53.3 | 43.6 | 67.5 | 47.2 | 48M | 301G |
| DuoMamba-S | 49.7 | 71.7 | 54.6 | 44.5 | 68.7 | 47.7 | 61M | 369G |
| DuoMamba-B | 50.6 | 72.5 | 55.6 | 45.2 | 69.6 | 48.8 | 111M | 564G |

**3× schedule (36 epochs) with multi-scale training**

| Backbone | AP<sup>b</sup> | AP<sup>b</sup><sub>50</sub> | AP<sup>b</sup><sub>75</sub> | AP<sup>m</sup> | AP<sup>m</sup><sub>50</sub> | AP<sup>m</sup><sub>75</sub> | #Params | FLOPs |
|----------|------|---------|---------|------|---------|---------|---------|-------|
| DuoMamba-T | 49.9 | 71.3 | 54.7 | 44.3 | 68.5 | 47.8 | 48M | 301G |
| DuoMamba-S | 51.1 | 72.3 | 56.0 | 45.3 | 69.5 | 49.1 | 61M | 369G |

### ADE20K semantic segmentation

UPerNet, 512 × 512 training crops, 160k iterations, total batch size 16. SS/MS denote single-scale/multi-scale evaluation; reported FLOPs follow the manuscript's 2048 × 512 convention.

| Backbone | mIoU (SS) | mIoU (MS) | #Params | FLOPs |
|----------|-----------|-----------|---------|-------|
| DuoMamba-T | 49.5 | 50.0 | 57M | 978G |
| DuoMamba-S | 50.3 | 51.3 | 70M | 1048G |
| DuoMamba-B | 52.1 | 52.7 | 122M | 1250G |

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
