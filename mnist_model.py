from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .losses import kl_unit_gaussian
from .ops import make_conv2d, make_deconv2d, make_linear


def _pixcnn_gate(inputs: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inputs[:, 0::2]) * torch.sigmoid(inputs[:, 1::2])


class _ConditionalGate(nn.Module):
    def __init__(self, latent_dim: int, dim: int) -> None:
        super().__init__()
        self.to_tanh = make_linear(latent_dim, dim, weightnorm=True)
        self.to_sigmoid = make_linear(latent_dim, dim, weightnorm=True)

    def forward(self, inputs: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        a = inputs[:, 0::2] + self.to_tanh(latents).unsqueeze(-1).unsqueeze(-1)
        b = inputs[:, 1::2] + self.to_sigmoid(latents).unsqueeze(-1).unsqueeze(-1)
        return torch.tanh(a) * torch.sigmoid(b)


class _MNISTPixelCNNLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_pix: int,
        filter_size: int,
        n_channels: int,
        *,
        latent_dim: int | None = None,
        conditioned: bool = False,
        hstack: str = "hstack",
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.conditioned = conditioned
        self.residual = residual
        self.vstack = make_conv2d(
            input_dim,
            2 * dim_pix,
            filter_size,
            mask_type="vstack",
            mask_n_channels=n_channels,
            weightnorm=True,
        )
        self.v2h = make_conv2d(2 * dim_pix, 2 * dim_pix, 1, weightnorm=True)
        self.hstack = make_conv2d(
            input_dim,
            2 * dim_pix,
            (1, filter_size),
            mask_type=hstack,
            mask_n_channels=n_channels,
            weightnorm=True,
        )
        self.h2h = make_conv2d(dim_pix, dim_pix, 1, weightnorm=True)
        if conditioned:
            if latent_dim is None:
                raise ValueError("conditioned PixelCNN layers require latent_dim")
            self.v_gate = _ConditionalGate(latent_dim, dim_pix)
            self.h_gate = _ConditionalGate(latent_dim, dim_pix)

    def forward(
        self,
        x_v: torch.Tensor,
        x_h: torch.Tensor,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_pad = torch.zeros(x_v.size(0), x_v.size(1), 1, x_v.size(3), device=x_v.device, dtype=x_v.dtype)
        x_v_padded = torch.cat([zero_pad, x_v], dim=2)

        x_v_next = self.vstack(x_v_padded)
        if self.conditioned:
            x_v_next_gated = self.v_gate(x_v_next, latents)
        else:
            x_v_next_gated = _pixcnn_gate(x_v_next)

        x_v2h = self.v2h(x_v_next[:, :, :-1, :])
        x_h_next = self.hstack(x_h)
        if self.conditioned:
            x_h_next = self.h_gate(x_h_next + x_v2h, latents)
        else:
            x_h_next = _pixcnn_gate(x_h_next + x_v2h)
        x_h_next = self.h2h(x_h_next)

        if self.residual:
            x_h_next = x_h_next + x_h

        return x_v_next_gated[:, :, 1:, :], x_h_next


class _MNISTEncoder(nn.Module):
    def __init__(self, latent_dim: int, n_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = make_conv2d(n_channels, 32, 3, weightnorm=True)
        self.conv2 = make_conv2d(32, 32, 3, stride=2, weightnorm=True)
        self.conv3 = make_conv2d(32, 32, 3, weightnorm=True)
        self.conv4 = make_conv2d(32, 64, 3, stride=2, weightnorm=True)
        self.conv5 = make_conv2d(64, 64, 3, weightnorm=True)
        self.conv6 = make_conv2d(64, 64, 3, stride=2, weightnorm=True)
        self.conv7 = make_conv2d(64, 64, 3, weightnorm=True)
        self.conv8 = make_conv2d(64, 64, 3, weightnorm=True)
        self.out = make_linear(4 * 4 * 64, 2 * latent_dim, weightnorm=True)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = F.relu(self.conv1(inputs))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.conv4(output))
        output = F.pad(output, (0, 1, 0, 1))
        output = F.relu(self.conv5(output))
        output = F.relu(self.conv6(output))
        output = F.relu(self.conv7(output))
        output = F.relu(self.conv8(output))
        output = output.flatten(start_dim=1)
        output = self.out(output)
        return output[:, 0::2], output[:, 1::2]


class _MNISTUpsampleDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        dim_pix: int,
        filter_size: int,
        num_pixel_cnn_layer: int,
        n_channels: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.dim_pix = dim_pix
        self.fc = make_linear(latent_dim, 4 * 4 * 64, weightnorm=True)
        self.conv1 = make_conv2d(64, 64, 3, weightnorm=True)
        self.conv2 = make_conv2d(64, 64, 3, weightnorm=True)
        self.deconv3 = make_deconv2d(64, 64, 3, weightnorm=True)
        self.conv4 = make_conv2d(64, 64, 3, weightnorm=True)
        self.deconv5 = make_deconv2d(64, 32, 3, weightnorm=True)
        self.conv6 = make_conv2d(32, 32, 3, weightnorm=True)
        self.deconv7 = make_deconv2d(32, 32, 3, weightnorm=True)
        self.conv8 = make_conv2d(32, 32, 3, weightnorm=True)

        self.input_stack = _MNISTPixelCNNLayer(
            input_dim=n_channels + 32,
            dim_pix=dim_pix,
            filter_size=7,
            n_channels=n_channels,
            hstack="hstack_a",
            residual=False,
        )
        self.stacks = nn.ModuleList(
            [
                _MNISTPixelCNNLayer(
                    input_dim=dim_pix,
                    dim_pix=dim_pix,
                    filter_size=filter_size,
                    n_channels=n_channels,
                )
                for _ in range(num_pixel_cnn_layer)
            ]
        )
        self.out1 = make_conv2d(dim_pix, 2 * 32, 1, weightnorm=True)
        self.out2 = make_conv2d(32, 2 * 32, 1, weightnorm=True)
        self.out3 = make_conv2d(32, n_channels, 1, he_init=False, weightnorm=True)

    def forward(self, latents: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        output = self.fc(latents).view(latents.size(0), 64, 4, 4)
        output = F.relu(output)
        output = F.relu(self.conv1(output))
        output = F.relu(self.conv2(output))
        output = F.relu(self.deconv3(output))
        output = F.relu(self.conv4(output))
        output = output[:, :, :7, :7]
        output = F.relu(self.deconv5(output))
        output = F.relu(self.conv6(output))
        output = F.relu(self.deconv7(output))
        output = F.relu(self.conv8(output))

        images_with_latent = torch.cat([images, output], dim=1)
        x_v, x_h = self.input_stack(images_with_latent, images_with_latent)
        for layer in self.stacks:
            x_v, x_h = layer(x_v, x_h)

        output = _pixcnn_gate(self.out1(x_h))
        output = _pixcnn_gate(self.out2(output))
        return self.out3(output)


class _MNISTCondZBiasDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        dim_pix: int,
        filter_size: int,
        num_pixel_cnn_layer: int,
        n_channels: int = 1,
    ) -> None:
        super().__init__()
        self.input_stack = _MNISTPixelCNNLayer(
            input_dim=n_channels,
            dim_pix=dim_pix,
            filter_size=7,
            n_channels=n_channels,
            latent_dim=latent_dim,
            conditioned=True,
            hstack="hstack_a",
            residual=False,
        )
        self.stacks = nn.ModuleList(
            [
                _MNISTPixelCNNLayer(
                    input_dim=dim_pix,
                    dim_pix=dim_pix,
                    filter_size=filter_size,
                    n_channels=n_channels,
                    latent_dim=latent_dim,
                    conditioned=True,
                )
                for _ in range(num_pixel_cnn_layer)
            ]
        )
        self.out1 = make_conv2d(dim_pix, 2 * dim_pix, 1, weightnorm=True)
        self.out1_gate = _ConditionalGate(latent_dim, dim_pix)
        self.out2 = make_conv2d(dim_pix, 2 * dim_pix, 1, weightnorm=True)
        self.out2_gate = _ConditionalGate(latent_dim, dim_pix)
        self.out3 = make_conv2d(dim_pix, n_channels, 1, he_init=False, weightnorm=True)

    def forward(self, latents: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        x_v, x_h = self.input_stack(images, images, latents)
        for layer in self.stacks:
            x_v, x_h = layer(x_v, x_h, latents)
        output = self.out1_gate(self.out1(x_h), latents)
        output = self.out2_gate(self.out2(output), latents)
        return self.out3(output)


@dataclass
class MNISTForwardOutput:
    loss: torch.Tensor
    reconst: torch.Tensor
    reg: torch.Tensor
    alpha: float
    mu: torch.Tensor
    log_sigma: torch.Tensor
    latents: torch.Tensor
    logits: torch.Tensor


class MNISTPixelVAE(nn.Module):
    def __init__(
        self,
        num_pixel_cnn_layer: int,
        decoder_algorithm: str,
        dim_pix: int = 32,
        filter_size: int = 5,
        latent_dim: int = 64,
        alpha_iters: int = 10000,
        n_channels: int = 1,
        height: int = 28,
        width: int = 28,
    ) -> None:
        super().__init__()
        if decoder_algorithm not in {"cond_z_bias", "upsample_z_conv"}:
            raise ValueError("decoder_algorithm must be 'cond_z_bias' or 'upsample_z_conv'")
        self.decoder_algorithm = decoder_algorithm
        self.alpha_iters = alpha_iters
        self.n_channels = n_channels
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.encoder = _MNISTEncoder(latent_dim=latent_dim, n_channels=n_channels)
        if decoder_algorithm == "cond_z_bias":
            self.decoder = _MNISTCondZBiasDecoder(
                latent_dim=latent_dim,
                dim_pix=dim_pix,
                filter_size=filter_size,
                num_pixel_cnn_layer=num_pixel_cnn_layer,
                n_channels=n_channels,
            )
        else:
            self.decoder = _MNISTUpsampleDecoder(
                latent_dim=latent_dim,
                dim_pix=dim_pix,
                filter_size=filter_size,
                num_pixel_cnn_layer=num_pixel_cnn_layer,
                n_channels=n_channels,
            )

    def forward(self, images: torch.Tensor, total_iters: int = 0, sample: bool = True) -> MNISTForwardOutput:
        images = images.float()
        mu, log_sigma = self.encoder(images)
        if sample:
            latents = mu + torch.randn_like(mu) * torch.exp(log_sigma)
        else:
            latents = mu
        logits = self.decoder(latents, images)
        reconst = F.binary_cross_entropy_with_logits(logits, images, reduction="none").sum(dim=(1, 2, 3)).mean()
        reg = kl_unit_gaussian(mu, log_sigma).sum(dim=1).mean()
        alpha = min(1.0, float(total_iters) / float(self.alpha_iters))
        loss = reconst + (alpha * reg)
        return MNISTForwardOutput(
            loss=loss,
            reconst=reconst,
            reg=reg,
            alpha=alpha,
            mu=mu,
            log_sigma=log_sigma,
            latents=latents,
            logits=logits,
        )

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        latents = torch.randn(num_samples, self.latent_dim, device=device)
        probs = torch.zeros(num_samples, self.n_channels, self.height, self.width, device=device)
        samples = probs.clone()
        for row in range(self.height):
            for col in range(self.width):
                for channel in range(self.n_channels):
                    logits = self.decoder(latents, samples)
                    current_probs = torch.sigmoid(logits)
                    draw = torch.bernoulli(current_probs[:, channel, row, col])
                    samples[:, channel, row, col] = draw
                    probs[:, channel, row, col] = current_probs[:, channel, row, col]
        return probs
