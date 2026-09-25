# Installation

## Environment

Use **Linux x86-64**, **Python 3.10**, and an NVIDIA GPU. The CUDA/Triton scan kernels do not provide a supported macOS or CPU training path. A GPU supporting bfloat16 is needed for the manuscript's classification recipe.

The following versions are an **installation target**, not a verified training-environment lockfile. Configuration loading and Python/shell syntax were checked during release preparation; CUDA installation, training, and evaluation still require validation on a GPU machine.

| Component | Target version |
|---|---|
| Python | 3.10 |
| PyTorch / torchvision | 2.2.0 / 0.17.0 |
| PyTorch CUDA build | 11.8 |
| Triton | 2.2.0 |
| Mamba SSM / causal-conv1d | 2.0.4 / 1.4.0 |
| timm | 0.4.12 |
| MMEngine / MMCV | 0.10.7 / 2.1.0 |
| MMDetection | 3.3.0 |
| MMSegmentation | 1.2.2 |

PyTorch is pinned to match the submitted Triton version. The OpenMMLab versions are bounded to avoid installing MMCV 2.2, which is outside the supported range of the selected detection and segmentation releases. The segmentation package is named **`mmsegmentation`** on PyPI.

## Classification dependencies

Run from the repository root in a fresh environment:

```bash
conda create -n duomamba python=3.10 -y
conda activate duomamba
python -m pip install --upgrade pip
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install packaging ninja wheel 'setuptools<70'
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

Mamba and causal-conv1d may compile CUDA extensions. Install a matching CUDA development toolkit with `nvcc` available when a compatible binary wheel is unavailable. A CUDA runtime-only image is insufficient for source builds.

## Detection and segmentation dependencies

Install classification dependencies first. Install MMCV **with compiled operations**, not `mmcv-lite`. With the pinned PyTorch 2.2 target, MMCV 2.1 may need to be built from source:

```bash
MMCV_WITH_OPS=1 python -m pip install --no-build-isolation --no-binary mmcv mmcv==2.1.0
# Install only the task(s) you need:
python -m pip install -r requirements/detection.txt
python -m pip install -r requirements/segmentation.txt
python -m pip check
python -c "import mmcv.ops; print('MMCV CUDA operations imported')"
```

The two task adapters import their own framework independently; detection does not require MMSegmentation, and segmentation does not require MMDetection.

## Checks

A fast CUDA check, without datasets or checkpoints:

```bash
python scripts/smoke_test.py --variant tiny
```

This runs a forward pass and a backward pass with random input. It verifies basic execution, not checkpoint loading or model accuracy. Repeat with `--variant small` or `--variant base` as needed.

Configuration checks can run without PyTorch or CUDA in a separate environment:

```bash
python -m pip install -r requirements/checks.txt
python scripts/check_configs.py
```

## Sources for the installation target

- [PyTorch previous-version installation commands](https://pytorch.org/get-started/previous-versions/)
- [Mamba 2.0.4 installation requirements](https://github.com/state-spaces/mamba/tree/v2.0.4)
- [MMDetection 3.3.0 version constraints](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/__init__.py)
- [MMSegmentation 1.2.2 version constraints](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/__init__.py)
