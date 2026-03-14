from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PixelVAEPytorch import PixelVAE, get_pixelvae_config

PACKAGE_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instantiate or sample from the translated PixelVAE architecture.")
    parser.add_argument("--settings", default="mnist_256")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--sample", default=0, type=int, help="If > 0, run ancestral sampling and save the result tensor.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(PACKAGE_DIR / "runs" / "pixelvae" / "sample.pt"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = get_pixelvae_config(args.settings)
    model = PixelVAE(cfg).to(args.device)

    images = torch.randint(
        low=0,
        high=256,
        size=(args.batch_size, cfg.n_channels, cfg.height, cfg.width),
        device=args.device,
    )
    output = model(images, total_iters=1)
    print(
        f"settings={cfg.settings} mode={cfg.mode} logits_shape={tuple(output.logits.shape)} "
        f"loss={output.loss.item():.4f} reconst={output.reconst.item():.4f}"
    )

    if args.sample > 0:
        samples = model.sample(args.sample, device=torch.device(args.device)).cpu()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"samples": samples, "settings": cfg.settings}, out_path)
        print(f"saved_samples={out_path}")


if __name__ == "__main__":
    main()
