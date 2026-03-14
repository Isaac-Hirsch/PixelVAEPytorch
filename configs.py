from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PixelVAEConfig:
    dataset: str
    settings: str
    mode: str
    embed_inputs: bool
    pixel_level_pixcnn: bool
    higher_level_pixcnn: bool
    dim_embed: int
    dim_pix_1: int
    dim_1: int
    dim_2: int
    dim_3: int
    dim_4: int
    dim_pix_2: Optional[int]
    latent_dim_2: int
    alpha1_iters: int
    alpha2_iters: int
    kl_penalty: float
    beta_iters: int
    pix_2_n_blocks: int
    lr: float
    lr_decay_after: int
    lr_decay_factor: float
    batch_size: int
    n_channels: int
    height: int
    width: int
    latent_dim_1: int = 64
    latents1_height: int = 7
    latents1_width: int = 7
    dim_0: Optional[int] = None


_CONFIGS = {
    "mnist_256": PixelVAEConfig(
        dataset="mnist_256",
        settings="mnist_256",
        mode="one_level",
        embed_inputs=True,
        pixel_level_pixcnn=True,
        higher_level_pixcnn=True,
        dim_embed=16,
        dim_pix_1=32,
        dim_1=16,
        dim_2=32,
        dim_3=32,
        dim_4=64,
        dim_pix_2=None,
        latent_dim_2=128,
        alpha1_iters=5000,
        alpha2_iters=5000,
        kl_penalty=1.0,
        beta_iters=1000,
        pix_2_n_blocks=1,
        lr=1e-3,
        lr_decay_after=250000,
        lr_decay_factor=1.0,
        batch_size=100,
        n_channels=1,
        height=28,
        width=28,
        latent_dim_1=64,
        latents1_height=7,
        latents1_width=7,
    ),
    "32px_small": PixelVAEConfig(
        dataset="lsun_32",
        settings="32px_small",
        mode="two_level",
        embed_inputs=True,
        pixel_level_pixcnn=True,
        higher_level_pixcnn=True,
        dim_embed=16,
        dim_pix_1=128,
        dim_1=64,
        dim_2=128,
        dim_3=256,
        dim_4=512,
        dim_pix_2=512,
        latent_dim_1=64,
        latent_dim_2=512,
        alpha1_iters=2000,
        alpha2_iters=5000,
        kl_penalty=1.0,
        beta_iters=1000,
        pix_2_n_blocks=1,
        lr=1e-3,
        lr_decay_after=180000,
        lr_decay_factor=0.1,
        batch_size=64,
        n_channels=3,
        height=32,
        width=32,
        latents1_height=8,
        latents1_width=8,
    ),
    "32px_big": PixelVAEConfig(
        dataset="lsun_32",
        settings="32px_big",
        mode="two_level",
        embed_inputs=False,
        pixel_level_pixcnn=True,
        higher_level_pixcnn=True,
        dim_embed=16,
        dim_pix_1=256,
        dim_1=128,
        dim_2=256,
        dim_3=512,
        dim_4=512,
        dim_pix_2=512,
        latent_dim_1=128,
        latent_dim_2=512,
        alpha1_iters=2000,
        alpha2_iters=5000,
        kl_penalty=1.0,
        beta_iters=1000,
        pix_2_n_blocks=1,
        lr=1e-3,
        lr_decay_after=300000,
        lr_decay_factor=0.1,
        batch_size=64,
        n_channels=3,
        height=32,
        width=32,
        latents1_height=8,
        latents1_width=8,
    ),
    "64px_small": PixelVAEConfig(
        dataset="lsun_64",
        settings="64px_small",
        mode="two_level",
        embed_inputs=True,
        pixel_level_pixcnn=True,
        higher_level_pixcnn=True,
        dim_embed=16,
        dim_pix_1=128,
        dim_0=64,
        dim_1=64,
        dim_2=128,
        dim_3=256,
        dim_4=512,
        dim_pix_2=256,
        latent_dim_1=64,
        latent_dim_2=512,
        alpha1_iters=2000,
        alpha2_iters=10000,
        kl_penalty=1.0,
        beta_iters=1000,
        pix_2_n_blocks=1,
        lr=1e-3,
        lr_decay_after=180000,
        lr_decay_factor=0.1,
        batch_size=64,
        n_channels=3,
        height=64,
        width=64,
        latents1_height=16,
        latents1_width=16,
    ),
    "64px_big": PixelVAEConfig(
        dataset="imagenet_64",
        settings="64px_big",
        mode="two_level",
        embed_inputs=True,
        pixel_level_pixcnn=True,
        higher_level_pixcnn=True,
        dim_embed=16,
        dim_pix_1=384,
        dim_0=192,
        dim_1=256,
        dim_2=512,
        dim_3=512,
        dim_4=512,
        dim_pix_2=512,
        latent_dim_1=64,
        latent_dim_2=512,
        alpha1_iters=1000,
        alpha2_iters=10000,
        kl_penalty=1.0,
        beta_iters=500,
        pix_2_n_blocks=1,
        lr=1e-3,
        lr_decay_after=180000,
        lr_decay_factor=0.5,
        batch_size=48,
        n_channels=3,
        height=64,
        width=64,
        latents1_height=16,
        latents1_width=16,
    ),
    "64px_big_onelevel": PixelVAEConfig(
        dataset="imagenet_64",
        settings="64px_big_onelevel",
        mode="one_level",
        embed_inputs=True,
        pixel_level_pixcnn=True,
        higher_level_pixcnn=True,
        dim_embed=16,
        dim_pix_1=384,
        dim_0=192,
        dim_1=256,
        dim_2=512,
        dim_3=512,
        dim_4=512,
        dim_pix_2=512,
        latent_dim_1=64,
        latent_dim_2=512,
        alpha1_iters=50000,
        alpha2_iters=50000,
        kl_penalty=1.0,
        beta_iters=1000,
        pix_2_n_blocks=1,
        lr=1e-3,
        lr_decay_after=180000,
        lr_decay_factor=0.5,
        batch_size=48,
        n_channels=3,
        height=64,
        width=64,
        latents1_height=7,
        latents1_width=7,
    ),
}


def get_pixelvae_config(settings: str = "mnist_256") -> PixelVAEConfig:
    if settings not in _CONFIGS:
        valid = ", ".join(sorted(_CONFIGS))
        raise ValueError(f"Unknown PixelVAE settings '{settings}'. Expected one of: {valid}")
    return _CONFIGS[settings]
