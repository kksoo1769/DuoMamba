"""Run a dataset-free DuoMamba forward/backward check on CUDA."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("tiny", "small", "base"), default="tiny")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        parser.error("An NVIDIA GPU with a working CUDA installation is required.")
    if not torch.cuda.is_bf16_supported():
        parser.error("This check uses the paper's bfloat16 recipe; a supported GPU is required.")

    sys.path.insert(0, str(ROOT / "classification"))
    from config import get_config
    from models import build_model
    from models.duomamba import DuoMamba, DuoScanMixer

    assert DuoMamba.__name__ == 'DuoMamba' and DuoScanMixer.__name__ == 'DuoScanMixer'
    config = get_config(SimpleNamespace(
        cfg=str(ROOT / f"classification/configs/duomamba/{args.variant}.yaml"), opts=None))
    model = build_model(config)
    model = model.cuda().train()
    inputs = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(inputs)
        loss = output.float().square().mean()
    assert output.shape == (2, 1000), output.shape
    assert torch.isfinite(output).all(), "Non-finite model output"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "Non-finite or missing gradients"
    torch.cuda.synchronize()
    print(f"PASS: DuoMamba-{args.variant}, "
          f"forward/backward, output {tuple(output.shape)}")


if __name__ == "__main__":
    main()
