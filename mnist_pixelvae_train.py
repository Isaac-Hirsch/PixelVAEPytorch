from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PixelVAEPytorch import MNISTPixelVAE
from PixelVAEPytorch.mnist_data import load_mnist_dataset

PACKAGE_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PyTorch translation of the original MNIST PixelVAE.")
    parser.add_argument("-L", "--num_pixel_cnn_layer", required=True, type=int)
    parser.add_argument("-algo", "--decoder_algorithm", required=True, choices=["cond_z_bias", "upsample_z_conv"])
    parser.add_argument("-dpx", "--dim_pix", default=32, type=int)
    parser.add_argument("-fs", "--filter_size", default=5, type=int)
    parser.add_argument("-ldim", "--latent_dim", default=64, type=int)
    parser.add_argument("-ait", "--alpha_iters", default=10000, type=int)
    parser.add_argument("-o", "--out_dir", default=str(PACKAGE_DIR / "runs" / "mnist"))
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--steps", default=1000, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", default=500, type=int)
    parser.add_argument("--data-root", default=str(PACKAGE_DIR / "data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--fake-data", action="store_true", help="Use random binarized tensors instead of loading MNIST.")
    return parser


def _load_mnist(batch_size: int, data_root: str, download: bool, fake_data: bool) -> DataLoader:
    if fake_data:
        images = torch.bernoulli(torch.full((batch_size * 8, 1, 28, 28), 0.5))
        labels = torch.zeros(batch_size * 8, dtype=torch.long)
        dataset = TensorDataset(images, labels)
    else:
        dataset = load_mnist_dataset(root=data_root, train=True, download=download)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MNISTPixelVAE(
        num_pixel_cnn_layer=args.num_pixel_cnn_layer,
        decoder_algorithm=args.decoder_algorithm,
        dim_pix=args.dim_pix,
        filter_size=args.filter_size,
        latent_dim=args.latent_dim,
        alpha_iters=min(args.alpha_iters, args.steps),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _load_mnist(
        batch_size=args.batch_size,
        data_root=args.data_root,
        download=not args.no_download,
        fake_data=args.fake_data,
    )
    iterator = itertools.cycle(loader)

    def save_checkpoint(step: int, filename: str) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": step,
        }
        torch.save(checkpoint, out_dir / filename)

    for step in range(args.steps):
        images, _ = next(iterator)
        images = torch.bernoulli(images.to(device))
        optimizer.zero_grad(set_to_none=True)
        output = model(images, total_iters=step)
        output.loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(
                f"step={step} loss={output.loss.item():.4f} "
                f"reconst={output.reconst.item():.4f} reg={output.reg.item():.4f} alpha={output.alpha:.4f}"
            )
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(step, f"checkpoint_step_{step}.pt")

    final_step = max(args.steps - 1, 0)
    save_checkpoint(final_step, "checkpoint_final.pt")
    print(f"saved_final_checkpoint={out_dir / 'checkpoint_final.pt'}")


if __name__ == "__main__":
    main()
