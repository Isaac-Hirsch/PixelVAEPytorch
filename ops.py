from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _pair(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        first, second = tuple(value)
        return int(first), int(second)
    return int(value), int(value)


def _maybe_weightnorm(module: nn.Module, enabled: bool) -> nn.Module:
    if enabled:
        return nn.utils.parametrizations.weight_norm(module)
    return module


def _uniform_(tensor: torch.Tensor, stdev: float, gain: float = 1.0) -> None:
    bound = math.sqrt(3.0) * stdev * gain
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def init_conv2d(
    module: nn.Conv2d | nn.ConvTranspose2d,
    he_init: bool = True,
    mask_type: str | None = None,
    gain: float = 1.0,
) -> None:
    if isinstance(module, nn.ConvTranspose2d):
        kernel_h, kernel_w = _pair(module.kernel_size)
        fan_in = module.in_channels * kernel_h * kernel_w
        fan_out = module.out_channels * kernel_h * kernel_w
        stdev = math.sqrt(1.0 / fan_in)
        stdev *= 2.0
        if he_init:
            stdev *= math.sqrt(2.0)
    else:
        kernel_h, kernel_w = _pair(module.kernel_size)
        stride_h, stride_w = _pair(module.stride)
        fan_in = module.in_channels * kernel_h * kernel_w
        fan_out = module.out_channels * kernel_h * kernel_w / float(stride_h * stride_w)
        if mask_type is not None:
            fan_in /= 2.0
            fan_out /= 2.0
        if he_init:
            stdev = math.sqrt(4.0 / (fan_in + fan_out))
        else:
            stdev = math.sqrt(2.0 / (fan_in + fan_out))
    _uniform_(module.weight, stdev=stdev, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_linear(module: nn.Linear, initialization: str | None = None, gain: float = 1.0) -> None:
    input_dim = module.in_features
    output_dim = module.out_features
    if initialization in (None, "lecun"):
        stdev = math.sqrt(1.0 / input_dim)
        _uniform_(module.weight, stdev=stdev, gain=gain)
    elif initialization == "glorot":
        stdev = math.sqrt(2.0 / (input_dim + output_dim))
        _uniform_(module.weight, stdev=stdev, gain=gain)
    elif initialization == "he":
        stdev = math.sqrt(2.0 / input_dim)
        _uniform_(module.weight, stdev=stdev, gain=gain)
    elif initialization == "glorot_he":
        stdev = math.sqrt(4.0 / (input_dim + output_dim))
        _uniform_(module.weight, stdev=stdev, gain=gain)
    elif initialization == "orthogonal":
        nn.init.orthogonal_(module.weight, gain=gain)
    else:
        raise ValueError(f"Unsupported linear initialization '{initialization}'")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        mask_type: str,
        mask_n_channels: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        bias: bool = True,
    ) -> None:
        kernel_size = _pair(kernel_size)
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=_pair(stride),
            padding=_pair(padding),
            bias=bias,
        )
        self.mask_type = mask_type
        self.mask_n_channels = mask_n_channels
        self.register_buffer("mask", self._build_mask())

    def _build_mask(self) -> torch.Tensor:
        mask = torch.ones_like(self.weight)
        _, _, kernel_h, kernel_w = self.weight.shape
        center_row = kernel_h // 2
        center_col = kernel_w // 2

        if center_row == 0:
            mask[:, :, :, center_col + 1 :] = 0
        elif center_col == 0:
            mask[:, :, center_row + 1 :, :] = 0
        else:
            mask[:, :, center_row + 1 :, :] = 0
            mask[:, :, center_row, center_col + 1 :] = 0

        if self.mask_type in {"a", "b", "hstack_a", "hstack"}:
            for in_group in range(self.mask_n_channels):
                for out_group in range(self.mask_n_channels):
                    block_current = (
                        self.mask_type in {"a", "hstack_a"} and in_group >= out_group
                    ) or (self.mask_type == "b" and in_group > out_group)
                    if block_current:
                        mask[
                            out_group :: self.mask_n_channels,
                            in_group :: self.mask_n_channels,
                            center_row,
                            center_col,
                        ] = 0

        if self.mask_type == "vstack":
            mask[:, :, center_row, :] = 1

        return mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.conv2d(
            inputs,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def make_conv2d(
    input_dim: int,
    output_dim: int,
    filter_size: int | tuple[int, int],
    *,
    he_init: bool = True,
    mask_type: str | None = None,
    mask_n_channels: int = 1,
    stride: int = 1,
    weightnorm: bool = True,
    bias: bool = True,
    gain: float = 1.0,
) -> nn.Module:
    kernel_size = _pair(filter_size)
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    if mask_type is None:
        module: nn.Module = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    else:
        module = MaskedConv2d(
            mask_type=mask_type,
            mask_n_channels=mask_n_channels,
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    init_conv2d(module, he_init=he_init, mask_type=mask_type, gain=gain)
    return _maybe_weightnorm(module, weightnorm)


def make_deconv2d(
    input_dim: int,
    output_dim: int,
    filter_size: int,
    *,
    he_init: bool = True,
    weightnorm: bool = True,
    bias: bool = True,
) -> nn.Module:
    module = nn.ConvTranspose2d(
        input_dim,
        output_dim,
        kernel_size=filter_size,
        stride=2,
        padding=filter_size // 2,
        output_padding=1,
        bias=bias,
    )
    init_conv2d(module, he_init=he_init)
    return _maybe_weightnorm(module, weightnorm)


def make_linear(
    input_dim: int,
    output_dim: int,
    *,
    initialization: str | None = None,
    weightnorm: bool = True,
    bias: bool = True,
    gain: float = 1.0,
) -> nn.Module:
    module = nn.Linear(input_dim, output_dim, bias=bias)
    init_linear(module, initialization=initialization, gain=gain)
    return _maybe_weightnorm(module, weightnorm)


class SubpixelConv2d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filter_size: int,
        *,
        he_init: bool = True,
        weightnorm: bool = False,
        mask_type: str | None = None,
        mask_n_channels: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = make_conv2d(
            input_dim=input_dim,
            output_dim=4 * output_dim,
            filter_size=filter_size,
            he_init=he_init,
            mask_type=mask_type,
            mask_n_channels=mask_n_channels,
            weightnorm=weightnorm,
            bias=bias,
        )
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(inputs))


class ImageEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(images.long())
        embedded = embedded.permute(0, 1, 4, 2, 3).contiguous()
        batch, channels, dim, height, width = embedded.shape
        return embedded.view(batch, channels * dim, height, width)
