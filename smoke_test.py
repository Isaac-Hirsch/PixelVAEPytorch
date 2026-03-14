from __future__ import annotations

from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PixelVAEPytorch import MNISTPixelVAE, PixelVAE, get_pixelvae_config


def main() -> None:
    mnist = MNISTPixelVAE(
        num_pixel_cnn_layer=2,
        decoder_algorithm="cond_z_bias",
        dim_pix=16,
        filter_size=5,
        latent_dim=16,
    )
    mnist_images = torch.rand(4, 1, 28, 28)
    mnist_out = mnist(mnist_images, total_iters=10)
    print("mnist", tuple(mnist_out.logits.shape), f"{mnist_out.loss.item():.4f}")

    one_level_cfg = get_pixelvae_config("mnist_256")
    one_level_model = PixelVAE(one_level_cfg)
    one_level_images = torch.randint(0, 256, (2, 1, 28, 28))
    one_level_out = one_level_model(one_level_images, total_iters=10)
    print("one_level", tuple(one_level_out.logits.shape), f"{one_level_out.loss.item():.4f}")

    two_level_cfg = get_pixelvae_config("32px_small")
    two_level_model = PixelVAE(two_level_cfg)
    two_level_images = torch.randint(0, 256, (1, 3, 32, 32))
    two_level_out = two_level_model(two_level_images, total_iters=10)
    print("two_level", tuple(two_level_out.logits.shape), f"{two_level_out.loss.item():.4f}")


if __name__ == "__main__":
    main()
