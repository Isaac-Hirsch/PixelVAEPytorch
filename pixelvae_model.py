from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .configs import PixelVAEConfig
from .losses import kl_gaussian_gaussian, kl_unit_gaussian
from .ops import ImageEmbedding, SubpixelConv2d, make_conv2d, make_linear


def _elu(inputs: torch.Tensor) -> torch.Tensor:
    return F.elu(inputs)


def _pixcnn_gated_nonlinearity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(a) * torch.tanh(b)


def _split_logits(output: torch.Tensor, n_channels: int, height: int, width: int) -> torch.Tensor:
    batch = output.size(0)
    return output.view(batch, 256, n_channels, height, width).permute(0, 2, 3, 4, 1).contiguous()


def _softsign_split(mu_and_logsig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, logsig = torch.chunk(mu_and_logsig, 2, dim=1)
    sig = 0.5 * (F.softsign(logsig) + 1.0)
    logsig = torch.log(sig)
    return mu, logsig, sig


def _clamp_logsig_and_sig(
    logsig: torch.Tensor,
    sig: torch.Tensor,
    total_iters: int,
    beta_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    floor = 1.0 - min(1.0, float(total_iters) / float(beta_iters))
    floor_tensor = torch.tensor(floor, dtype=logsig.dtype, device=logsig.device)
    log_floor = torch.log(floor_tensor)
    return torch.maximum(logsig, log_floor), torch.maximum(sig, floor_tensor)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filter_size: int,
        *,
        mask_type: str | None = None,
        mask_n_channels: int = 1,
        resample: str | None = None,
        he_init: bool = True,
    ) -> None:
        super().__init__()
        if mask_type is not None and resample is not None:
            raise ValueError("masked residual blocks do not support resampling")
        self.mask_type = mask_type

        if resample == "down":
            self.shortcut = make_conv2d(input_dim, output_dim, 1, stride=2, he_init=False, weightnorm=False)
            self.conv1 = make_conv2d(input_dim, input_dim, filter_size, he_init=he_init, weightnorm=False)
            self.conv2 = make_conv2d(input_dim, output_dim, filter_size, stride=2, he_init=he_init, weightnorm=False, bias=False)
            self.bn = nn.BatchNorm2d(output_dim, eps=1e-2)
        elif resample == "up":
            self.shortcut = SubpixelConv2d(input_dim, output_dim, 1, he_init=False, weightnorm=False)
            self.conv1 = SubpixelConv2d(input_dim, output_dim, filter_size, he_init=he_init, weightnorm=False)
            self.conv2 = make_conv2d(output_dim, output_dim, filter_size, he_init=he_init, weightnorm=False, bias=False)
            self.bn = nn.BatchNorm2d(output_dim, eps=1e-2)
        elif mask_type is None:
            self.shortcut = nn.Identity() if input_dim == output_dim else make_conv2d(input_dim, output_dim, 1, he_init=False, weightnorm=False)
            self.conv1 = make_conv2d(input_dim, output_dim, filter_size, he_init=he_init, weightnorm=False)
            self.conv2 = make_conv2d(output_dim, output_dim, filter_size, he_init=he_init, weightnorm=False, bias=False)
            self.bn = nn.BatchNorm2d(output_dim, eps=1e-2)
        else:
            self.shortcut = nn.Identity() if input_dim == output_dim else make_conv2d(
                input_dim,
                output_dim,
                1,
                he_init=False,
                mask_type=mask_type,
                mask_n_channels=mask_n_channels,
            )
            self.conv1_a = make_conv2d(
                input_dim,
                output_dim,
                filter_size,
                he_init=he_init,
                mask_type=mask_type,
                mask_n_channels=mask_n_channels,
            )
            self.conv1_b = make_conv2d(
                input_dim,
                output_dim,
                filter_size,
                he_init=he_init,
                mask_type=mask_type,
                mask_n_channels=mask_n_channels,
            )
            self.conv2 = make_conv2d(
                output_dim,
                output_dim,
                filter_size,
                he_init=he_init,
                mask_type=mask_type,
                mask_n_channels=mask_n_channels,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(inputs)
        if self.mask_type is None:
            output = _elu(inputs)
            output = self.conv1(output)
            output = _elu(output)
            output = self.conv2(output)
            output = self.bn(output)
        else:
            output = _elu(inputs)
            output = _pixcnn_gated_nonlinearity(self.conv1_a(output), self.conv1_b(output))
            output = self.conv2(output)
        return shortcut + output


class Enc1(nn.Module):
    def __init__(self, cfg: PixelVAEConfig, input_channels: int) -> None:
        super().__init__()
        if cfg.width == 64:
            if cfg.embed_inputs:
                self.input_conv = make_conv2d(input_channels, cfg.dim_0, 1, he_init=False)
                self.input_res0 = ResidualBlock(cfg.dim_0, cfg.dim_0, 3)
                self.input_res = ResidualBlock(cfg.dim_0, cfg.dim_1, 3, resample="down")
            else:
                self.input_conv = make_conv2d(input_channels, cfg.dim_1, 1, he_init=False)
                self.input_res0 = None
                self.input_res = ResidualBlock(cfg.dim_1, cfg.dim_1, 3, resample="down")
        else:
            self.input_conv = make_conv2d(input_channels, cfg.dim_1, 1, he_init=False)
            self.input_res0 = None
            self.input_res = None

        self.res1pre = ResidualBlock(cfg.dim_1, cfg.dim_1, 3)
        self.res1pre2 = ResidualBlock(cfg.dim_1, cfg.dim_1, 3)
        self.res1 = ResidualBlock(cfg.dim_1, cfg.dim_2, 3, resample="down")

        self.use_16 = cfg.latents1_width == 16
        if self.use_16:
            self.res4pre = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res4 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res4post = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.out = make_conv2d(cfg.dim_2, 2 * cfg.latent_dim_1, 1, he_init=False)
        else:
            self.res2pre = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res2pre2 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res2 = ResidualBlock(cfg.dim_2, cfg.dim_3, 3, resample="down")
            self.res3pre = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
            self.res3pre2 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
            self.res3pre3 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
            self.out = make_conv2d(cfg.dim_3, 2 * cfg.latent_dim_1, 1, he_init=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.input_conv(images)
        if self.input_res0 is not None:
            output = self.input_res0(output)
        if self.input_res is not None:
            output = self.input_res(output)
        output = self.res1pre(output)
        output = self.res1pre2(output)
        output = self.res1(output)
        if self.use_16:
            output = self.res4pre(output)
            output = self.res4(output)
            output = self.res4post(output)
            return self.out(output), output
        output = self.res2pre(output)
        output = self.res2pre2(output)
        output = self.res2(output)
        output = self.res3pre(output)
        output = self.res3pre2(output)
        output = self.res3pre3(output)
        return self.out(output), output


class Dec1(nn.Module):
    def __init__(self, cfg: PixelVAEConfig, input_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_16 = cfg.latents1_width == 16
        if self.use_16:
            self.input_conv = make_conv2d(cfg.latent_dim_1, cfg.dim_2, 1, he_init=False)
            self.res1a = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res1b = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res1c = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
        else:
            self.input_conv = make_conv2d(cfg.latent_dim_1, cfg.dim_3, 1, he_init=False)
            self.res1 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
            self.res1post = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
            self.res1post2 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
            self.res2 = ResidualBlock(cfg.dim_3, cfg.dim_2, 3, resample="up")
            self.res2post = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res2post2 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)

        self.res3 = ResidualBlock(cfg.dim_2, cfg.dim_1, 3, resample="up")
        self.res3post = ResidualBlock(cfg.dim_1, cfg.dim_1, 3)
        self.res3post2 = ResidualBlock(cfg.dim_1, cfg.dim_1, 3)

        if cfg.width == 64:
            self.res4 = ResidualBlock(cfg.dim_1, cfg.dim_0, 3, resample="up")
            self.res4post = ResidualBlock(cfg.dim_0, cfg.dim_0, 3)

        if cfg.pixel_level_pixcnn:
            pix_input_dim = cfg.dim_0 if cfg.width == 64 else cfg.dim_1
            self.masked_images = make_conv2d(
                input_channels,
                pix_input_dim,
                5,
                mask_type="a",
                mask_n_channels=cfg.n_channels,
                he_init=False,
            )
            self.pix2 = ResidualBlock(2 * pix_input_dim, cfg.dim_pix_1, 3, mask_type="b", mask_n_channels=cfg.n_channels)
            self.pix3 = ResidualBlock(cfg.dim_pix_1, cfg.dim_pix_1, 3, mask_type="b", mask_n_channels=cfg.n_channels)
            self.pix4 = ResidualBlock(cfg.dim_pix_1, cfg.dim_pix_1, 3, mask_type="b", mask_n_channels=cfg.n_channels) if cfg.width == 64 else None
            self.out = make_conv2d(
                cfg.dim_pix_1,
                256 * cfg.n_channels,
                1,
                mask_type="b",
                mask_n_channels=cfg.n_channels,
                he_init=False,
            )
        else:
            last_dim = cfg.dim_0 if cfg.width == 64 else cfg.dim_1
            self.out = make_conv2d(last_dim, 256 * cfg.n_channels, 1, he_init=False)

    def forward(self, latents: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        output = torch.clamp(latents, -50.0, 50.0)
        output = self.input_conv(output)
        if self.use_16:
            output = self.res1a(output)
            output = self.res1b(output)
            output = self.res1c(output)
        else:
            output = self.res1(output)
            output = self.res1post(output)
            output = self.res1post2(output)
            output = self.res2(output)
            output = self.res2post(output)
            output = self.res2post2(output)
        output = self.res3(output)
        output = self.res3post(output)
        output = self.res3post2(output)
        if cfg.width == 64:
            output = self.res4(output)
            output = self.res4post(output)

        if cfg.pixel_level_pixcnn:
            masked_images = self.masked_images(images)
            output = output / 2.0
            output = torch.cat([masked_images, output], dim=1)
            output = self.pix2(output)
            output = self.pix3(output)
            if self.pix4 is not None:
                output = self.pix4(output)
        output = self.out(output)
        return _split_logits(output, cfg.n_channels, cfg.height, cfg.width)


class Enc2(nn.Module):
    def __init__(self, cfg: PixelVAEConfig) -> None:
        super().__init__()
        self.use_16 = cfg.latents1_width == 16
        if self.use_16:
            self.res0 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res1pre = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res1pre2 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res1 = ResidualBlock(cfg.dim_2, cfg.dim_3, 3, resample="down")
        self.res2pre = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
        self.res2pre2 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
        self.res2pre3 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
        self.res1a = ResidualBlock(cfg.dim_3, cfg.dim_4, 3, resample="down")
        self.res2prea = ResidualBlock(cfg.dim_4, cfg.dim_4, 3)
        self.res2 = ResidualBlock(cfg.dim_4, cfg.dim_4, 3)
        self.res2post = ResidualBlock(cfg.dim_4, cfg.dim_4, 3)
        self.out = make_linear(4 * 4 * cfg.dim_4, 2 * cfg.latent_dim_2)

    def forward(self, h1: torch.Tensor) -> torch.Tensor:
        output = h1
        if self.use_16:
            output = self.res0(output)
            output = self.res1pre(output)
            output = self.res1pre2(output)
            output = self.res1(output)
        output = self.res2pre(output)
        output = self.res2pre2(output)
        output = self.res2pre3(output)
        output = self.res1a(output)
        output = self.res2prea(output)
        output = self.res2(output)
        output = self.res2post(output)
        return self.out(output.flatten(start_dim=1))


class Dec2(nn.Module):
    def __init__(self, cfg: PixelVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_16 = cfg.latents1_width == 16
        self.input = make_linear(cfg.latent_dim_2, 4 * 4 * cfg.dim_4)
        self.res1pre = ResidualBlock(cfg.dim_4, cfg.dim_4, 3)
        self.res1 = ResidualBlock(cfg.dim_4, cfg.dim_4, 3)
        self.res1post = ResidualBlock(cfg.dim_4, cfg.dim_4, 3)
        self.res3 = ResidualBlock(cfg.dim_4, cfg.dim_3, 3, resample="up")
        self.res3post = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
        self.res3post2 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)
        self.res3post3 = ResidualBlock(cfg.dim_3, cfg.dim_3, 3)

        if self.use_16:
            self.res3post5 = ResidualBlock(cfg.dim_3, cfg.dim_2, 3, resample="up")
            self.res3post6 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res3post7 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)
            self.res3post8 = ResidualBlock(cfg.dim_2, cfg.dim_2, 3)

        if cfg.higher_level_pixcnn:
            dim = cfg.dim_2 if self.use_16 else cfg.dim_3
            self.masked_targets = make_conv2d(
                cfg.latent_dim_1,
                dim,
                5,
                mask_type="a",
                mask_n_channels=cfg.pix_2_n_blocks,
                he_init=False,
            )
            self.pix2 = ResidualBlock(2 * dim, cfg.dim_pix_2, 3, mask_type="b", mask_n_channels=cfg.pix_2_n_blocks)
            self.pix3 = ResidualBlock(cfg.dim_pix_2, cfg.dim_pix_2, 3, mask_type="b", mask_n_channels=cfg.pix_2_n_blocks)
            self.pix4 = ResidualBlock(cfg.dim_pix_2, cfg.dim_pix_2, 1, mask_type="b", mask_n_channels=cfg.pix_2_n_blocks)
            self.out = make_conv2d(
                cfg.dim_pix_2,
                2 * cfg.latent_dim_1,
                1,
                mask_type="b",
                mask_n_channels=cfg.pix_2_n_blocks,
                he_init=False,
            )
        else:
            dim = cfg.dim_2 if self.use_16 else cfg.dim_3
            self.out = make_conv2d(
                dim,
                2 * cfg.latent_dim_1,
                1,
                mask_type="b",
                mask_n_channels=cfg.pix_2_n_blocks,
                he_init=False,
            )

    def forward(self, latents: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        output = torch.clamp(latents, -50.0, 50.0)
        output = self.input(output).view(latents.size(0), cfg.dim_4, 4, 4)
        output = self.res1pre(output)
        output = self.res1(output)
        output = self.res1post(output)
        output = self.res3(output)
        output = self.res3post(output)
        output = self.res3post2(output)
        output = self.res3post3(output)
        if self.use_16:
            output = self.res3post5(output)
            output = self.res3post6(output)
            output = self.res3post7(output)
            output = self.res3post8(output)

        if cfg.higher_level_pixcnn:
            masked_targets = self.masked_targets(targets)
            output = output / 2.0
            output = torch.cat([masked_targets, output], dim=1)
            output = self.pix2(output)
            output = self.pix3(output)
            output = self.pix4(output)
        return self.out(output)


class EncFull(nn.Module):
    def __init__(self, cfg: PixelVAEConfig, input_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.width == 64:
            start_dim = cfg.dim_0
            self.input = make_conv2d(input_channels, cfg.dim_0, 1, he_init=False)
            self.blocks = nn.ModuleList(
                [
                    ResidualBlock(cfg.dim_0, cfg.dim_0, 3),
                    ResidualBlock(cfg.dim_0, cfg.dim_1, 3, resample="down"),
                    ResidualBlock(cfg.dim_1, cfg.dim_1, 3),
                    ResidualBlock(cfg.dim_1, cfg.dim_1, 3),
                    ResidualBlock(cfg.dim_1, cfg.dim_2, 3, resample="down"),
                    ResidualBlock(cfg.dim_2, cfg.dim_2, 3),
                    ResidualBlock(cfg.dim_2, cfg.dim_2, 3),
                    ResidualBlock(cfg.dim_2, cfg.dim_3, 3, resample="down"),
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                    ResidualBlock(cfg.dim_3, cfg.dim_4, 3, resample="down"),
                    ResidualBlock(cfg.dim_4, cfg.dim_4, 3),
                    ResidualBlock(cfg.dim_4, cfg.dim_4, 3),
                ]
            )
            self.out = make_linear(4 * 4 * cfg.dim_4, 2 * cfg.latent_dim_2, initialization="glorot")
            self.reduce_mean = False
        else:
            start_dim = cfg.dim_1
            self.input = make_conv2d(input_channels, start_dim, 1, he_init=False)
            self.blocks = nn.ModuleList(
                [
                    ResidualBlock(cfg.dim_1, cfg.dim_1, 3),
                    ResidualBlock(cfg.dim_1, cfg.dim_2, 3, resample="down"),
                    ResidualBlock(cfg.dim_2, cfg.dim_2, 3),
                    ResidualBlock(cfg.dim_2, cfg.dim_3, 3, resample="down"),
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                ]
            )
            self.out = make_linear(cfg.dim_3, 2 * cfg.latent_dim_2, initialization="glorot")
            self.reduce_mean = True
        self.start_dim = start_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.input(images)
        for block in self.blocks:
            output = block(output)
        if self.reduce_mean:
            output = output.mean(dim=(2, 3))
        else:
            output = output.flatten(start_dim=1)
        return self.out(output)


class DecFull(nn.Module):
    def __init__(self, cfg: PixelVAEConfig, input_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.width == 64:
            self.input = make_linear(cfg.latent_dim_2, 4 * 4 * cfg.dim_4, initialization="glorot")
            self.blocks = nn.ModuleList(
                [
                    ResidualBlock(cfg.dim_4, cfg.dim_4, 3),
                    ResidualBlock(cfg.dim_4, cfg.dim_4, 3),
                    ResidualBlock(cfg.dim_4, cfg.dim_3, 3, resample="up"),
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                    ResidualBlock(cfg.dim_3, cfg.dim_2, 3, resample="up"),
                    ResidualBlock(cfg.dim_2, cfg.dim_2, 3),
                    ResidualBlock(cfg.dim_2, cfg.dim_2, 3),
                    ResidualBlock(cfg.dim_2, cfg.dim_1, 3, resample="up"),
                    ResidualBlock(cfg.dim_1, cfg.dim_1, 3),
                    ResidualBlock(cfg.dim_1, cfg.dim_1, 3),
                    ResidualBlock(cfg.dim_1, cfg.dim_0, 3, resample="up"),
                    ResidualBlock(cfg.dim_0, cfg.dim_0, 3),
                ]
            )
            dim = cfg.dim_0
            self.tile_7x7 = False
        else:
            self.input = make_linear(cfg.latent_dim_2, cfg.dim_3, initialization="glorot")
            self.blocks = nn.ModuleList(
                [
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                    ResidualBlock(cfg.dim_3, cfg.dim_3, 3),
                    ResidualBlock(cfg.dim_3, cfg.dim_2, 3, resample="up"),
                    ResidualBlock(cfg.dim_2, cfg.dim_2, 3),
                    ResidualBlock(cfg.dim_2, cfg.dim_1, 3, resample="up"),
                    ResidualBlock(cfg.dim_1, cfg.dim_1, 3),
                ]
            )
            dim = cfg.dim_1
            self.tile_7x7 = True

        if cfg.pixel_level_pixcnn:
            self.masked_images = make_conv2d(
                input_channels,
                dim,
                5,
                mask_type="a",
                mask_n_channels=cfg.n_channels,
                he_init=False,
            )
            self.pix2 = ResidualBlock(2 * dim, cfg.dim_pix_1, 3, mask_type="b", mask_n_channels=cfg.n_channels)
            self.pix3 = ResidualBlock(cfg.dim_pix_1, cfg.dim_pix_1, 3, mask_type="b", mask_n_channels=cfg.n_channels)
            self.pix4 = ResidualBlock(cfg.dim_pix_1, cfg.dim_pix_1, 3, mask_type="b", mask_n_channels=cfg.n_channels)
            self.pix5 = ResidualBlock(cfg.dim_pix_1, cfg.dim_pix_1, 3, mask_type="b", mask_n_channels=cfg.n_channels) if cfg.width != 64 else None
            self.out = make_conv2d(
                cfg.dim_pix_1,
                256 * cfg.n_channels,
                1,
                mask_type="b",
                mask_n_channels=cfg.n_channels,
                he_init=False,
            )
        else:
            self.pix2 = None
            self.out = make_conv2d(dim, 256 * cfg.n_channels, 1, he_init=False)

    def forward(self, latents: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        output = torch.clamp(latents, -50.0, 50.0)
        output = self.input(output)
        if self.tile_7x7:
            output = output.view(output.size(0), cfg.dim_3, 1, 1).expand(-1, -1, 7, 7)
        else:
            output = output.view(output.size(0), cfg.dim_4, 4, 4)
        for block in self.blocks:
            output = block(output)

        if cfg.pixel_level_pixcnn:
            masked_images = self.masked_images(images)
            output = torch.cat([masked_images, output], dim=1)
            output = self.pix2(output)
            output = self.pix3(output)
            output = self.pix4(output)
            if self.pix5 is not None:
                output = self.pix5(output)
        output = self.out(output)
        return _split_logits(output, cfg.n_channels, cfg.height, cfg.width)


@dataclass
class PixelVAEForwardOutput:
    loss: torch.Tensor
    reconst: torch.Tensor
    logits: torch.Tensor
    alpha1: float
    kl1: torch.Tensor
    mu1: torch.Tensor
    logsig1: torch.Tensor
    sig1: torch.Tensor
    latents1: torch.Tensor
    alpha2: float | None = None
    kl2: torch.Tensor | None = None
    mu2: torch.Tensor | None = None
    logsig2: torch.Tensor | None = None
    sig2: torch.Tensor | None = None
    latents2: torch.Tensor | None = None
    mu1_prior: torch.Tensor | None = None
    logsig1_prior: torch.Tensor | None = None
    sig1_prior: torch.Tensor | None = None


class PixelVAE(nn.Module):
    def __init__(self, config: PixelVAEConfig) -> None:
        super().__init__()
        self.config = config
        input_channels = config.n_channels * config.dim_embed if config.embed_inputs else config.n_channels
        self.embedding = ImageEmbedding(256, config.dim_embed) if config.embed_inputs else None

        if config.mode == "one_level":
            self.enc_full = EncFull(config, input_channels)
            self.dec_full = DecFull(config, input_channels)
        elif config.mode == "two_level":
            self.enc1 = Enc1(config, input_channels)
            self.dec1 = Dec1(config, input_channels)
            self.enc2 = Enc2(config)
            self.dec2 = Dec2(config)
        else:
            raise ValueError(f"Unsupported PixelVAE mode '{config.mode}'")

    def _scale_images(self, images: torch.Tensor) -> torch.Tensor:
        return (images.float() - 128.0) / 64.0

    def _decoder_inputs(self, images: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            return self.embedding(images.long())
        return self._scale_images(images)

    def _reconstruction_loss(self, logits: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.reshape(-1, 256)
        flat_targets = images.reshape(-1).long()
        return F.cross_entropy(flat_logits, flat_targets, reduction="mean")

    def forward(self, images: torch.Tensor, total_iters: int = 0) -> PixelVAEForwardOutput:
        cfg = self.config
        images = images.long()
        decoder_inputs = self._decoder_inputs(images)

        if cfg.mode == "one_level":
            mu_and_logsig1 = self.enc_full(decoder_inputs)
            mu1, logsig1, sig1 = _softsign_split(mu_and_logsig1)
            latents1 = mu1 + torch.randn_like(mu1) * sig1
            logits = self.dec_full(latents1, decoder_inputs)
            reconst = self._reconstruction_loss(logits, images)
            alpha1 = min(1.0, float(total_iters + 1) / float(cfg.alpha1_iters)) * cfg.kl_penalty
            kl1 = kl_unit_gaussian(mu1, logsig1, sig1).mean()
            kl1 = kl1 * float(cfg.latent_dim_2) / float(cfg.n_channels * cfg.width * cfg.height)
            loss = reconst + (alpha1 * kl1)
            return PixelVAEForwardOutput(
                loss=loss,
                reconst=reconst,
                logits=logits,
                alpha1=alpha1,
                kl1=kl1,
                mu1=mu1,
                logsig1=logsig1,
                sig1=sig1,
                latents1=latents1,
            )

        mu_and_logsig1, h1 = self.enc1(decoder_inputs)
        mu1, logsig1, sig1 = _softsign_split(mu_and_logsig1)
        latents1 = mu1 + torch.randn_like(mu1) * sig1
        logits = self.dec1(latents1, decoder_inputs)
        reconst = self._reconstruction_loss(logits, images)

        mu_and_logsig2 = self.enc2(h1)
        mu2, logsig2, sig2 = _softsign_split(mu_and_logsig2)
        latents2 = mu2 + torch.randn_like(mu2) * sig2
        outputs2 = self.dec2(latents2, latents1)
        mu1_prior, logsig1_prior, sig1_prior = _softsign_split(outputs2)
        logsig1_prior, sig1_prior = _clamp_logsig_and_sig(logsig1_prior, sig1_prior, total_iters, cfg.beta_iters)
        mu1_prior = 2.0 * F.softsign(mu1_prior / 2.0)

        alpha1 = min(1.0, float(total_iters + 1) / float(cfg.alpha1_iters)) * cfg.kl_penalty
        alpha2 = min(1.0, float(total_iters + 1) / float(cfg.alpha2_iters)) * alpha1

        kl1 = kl_gaussian_gaussian(mu1, logsig1, sig1, mu1_prior, logsig1_prior, sig1_prior).mean()
        kl2 = kl_unit_gaussian(mu2, logsig2, sig2).mean()
        kl1 = kl1 * float(cfg.latent_dim_1 * cfg.latents1_width * cfg.latents1_height) / float(cfg.n_channels * cfg.width * cfg.height)
        kl2 = kl2 * float(cfg.latent_dim_2) / float(cfg.n_channels * cfg.width * cfg.height)
        loss = reconst + (alpha1 * kl1) + (alpha2 * kl2)

        return PixelVAEForwardOutput(
            loss=loss,
            reconst=reconst,
            logits=logits,
            alpha1=alpha1,
            kl1=kl1,
            mu1=mu1,
            logsig1=logsig1,
            sig1=sig1,
            latents1=latents1,
            alpha2=alpha2,
            kl2=kl2,
            mu2=mu2,
            logsig2=logsig2,
            sig2=sig2,
            latents2=latents2,
            mu1_prior=mu1_prior,
            logsig1_prior=logsig1_prior,
            sig1_prior=sig1_prior,
        )

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        cfg = self.config
        self.eval()

        if cfg.mode == "one_level":
            latents = torch.randn(num_samples, cfg.latent_dim_2, device=device)
            pixels = torch.zeros(num_samples, cfg.n_channels, cfg.height, cfg.width, dtype=torch.long, device=device)
            for row in range(cfg.height):
                for col in range(cfg.width):
                    for channel in range(cfg.n_channels):
                        logits = self.dec_full(latents, self._decoder_inputs(pixels))
                        probs = torch.softmax(logits[:, channel, row, col], dim=-1)
                        pixels[:, channel, row, col] = torch.multinomial(probs, 1).squeeze(-1)
            return pixels

        z2 = torch.randn(num_samples, cfg.latent_dim_2, device=device)
        z1 = torch.zeros(
            num_samples,
            cfg.latent_dim_1,
            cfg.latents1_height,
            cfg.latents1_width,
            device=device,
        )
        epsilon_1 = torch.randn_like(z1)
        for row in range(cfg.latents1_height):
            for col in range(cfg.latents1_width):
                prior = self.dec2(z2, z1)
                mu1_prior, logsig1_prior, sig1_prior = _softsign_split(prior)
                logsig1_prior, sig1_prior = _clamp_logsig_and_sig(logsig1_prior, sig1_prior, cfg.beta_iters, cfg.beta_iters)
                mu1_prior = 2.0 * F.softsign(mu1_prior / 2.0)
                z1[:, :, row, col] = mu1_prior[:, :, row, col] + sig1_prior[:, :, row, col] * epsilon_1[:, :, row, col]

        pixels = torch.zeros(num_samples, cfg.n_channels, cfg.height, cfg.width, dtype=torch.long, device=device)
        for row in range(cfg.height):
            for col in range(cfg.width):
                for channel in range(cfg.n_channels):
                    logits = self.dec1(z1, self._decoder_inputs(pixels))
                    probs = torch.softmax(logits[:, channel, row, col], dim=-1)
                    pixels[:, channel, row, col] = torch.multinomial(probs, 1).squeeze(-1)
        return pixels
