from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PixelVAEPytorch import MNISTPixelVAE

PACKAGE_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the PyTorch translation of the original MNIST PixelVAE.")
    parser.add_argument("-L", "--num_pixel_cnn_layer", required=True, type=int)
    parser.add_argument("-algo", "--decoder_algorithm", required=True, choices=["cond_z_bias", "upsample_z_conv"])
    parser.add_argument("-dpx", "--dim_pix", default=32, type=int)
    parser.add_argument("-fs", "--filter_size", default=5, type=int)
    parser.add_argument("-ldim", "--latent_dim", default=64, type=int)
    parser.add_argument("-ait", "--alpha_iters", default=10000, type=int)
    parser.add_argument("-w", "--pre_trained_weights", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", default=100, type=int)
    parser.add_argument("--out", default=str(PACKAGE_DIR / "runs" / "mnist" / "eval_samples.png"))
    return parser


def _save_samples_png(samples: torch.Tensor, out_path: Path) -> None:
    samples = samples.detach().cpu().clamp(0.0, 1.0)
    num_samples, n_channels, height, width = samples.shape

    rows = int(math.floor(math.sqrt(num_samples)))
    while rows > 1 and num_samples % rows != 0:
        rows -= 1
    cols = int(math.ceil(num_samples / rows))

    if n_channels == 1:
        grid = torch.zeros(rows * height, cols * width, dtype=torch.uint8)
        for index in range(num_samples):
            row = index // cols
            col = index % cols
            tile = (samples[index, 0] * 255.0).round().to(torch.uint8)
            grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
        image = Image.fromarray(grid.numpy(), mode="L")
    else:
        grid = torch.zeros(rows * height, cols * width, n_channels, dtype=torch.uint8)
        for index in range(num_samples):
            row = index // cols
            col = index % cols
            tile = (samples[index].permute(1, 2, 0) * 255.0).round().to(torch.uint8)
            grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
        image = Image.fromarray(grid.numpy(), mode="RGB")

    image.save(out_path)


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.pre_trained_weights)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass a real `.pt` file produced by `mnist_pixelvae_train.py`."
        )
    model = MNISTPixelVAE(
        num_pixel_cnn_layer=args.num_pixel_cnn_layer,
        decoder_algorithm=args.decoder_algorithm,
        dim_pix=args.dim_pix,
        filter_size=args.filter_size,
        latent_dim=args.latent_dim,
        alpha_iters=args.alpha_iters,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    samples = model.sample(args.num_samples, device=device).cpu()

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_samples_png(samples, out_path)
    print(f"saved_samples={out_path}")


if __name__ == "__main__":
    main()
