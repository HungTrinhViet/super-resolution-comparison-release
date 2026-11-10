"""FSRCNN(56,12,4) author-layout weights and the fixed RGB adapter."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as functional

from .matlab_color import (
    matlab_uint8_round, rgb2ycbcr_uint8, ycbcr2rgb_uint8,
)
from .matlab_resize import imresize_bicubic_float

CHECKPOINT = Path('data/quarantine/fsrcnn/yaBorn_c6b50c5_FSRCNN_x4.mat')
CHECKPOINT_SHA256 = 'af005f2ba6e90ef45f1807045c3f78132e15b57d9056e4691c79e34863dbdd4f'
SOURCE_NOTICE = (
    'Bản đăng lại của gói FSRCNN_test (Chao Dong, ECCV 2016); '
    'hai bản sao độc lập trùng SHA-256; nguồn gốc trực tiếp từ CUHK và '
    'cách chọn checkpoint chưa xác minh; chỉ dùng học thuật.'
)
WEIGHT_SHAPES = (
    (1, 25, 56), (56, 1, 12), *((12, 9, 12),)*4,
    (12, 1, 56), (1, 81, 56),
)


def convert_weights(arrays: dict[str, NDArray]) -> dict[str, torch.Tensor]:
    """Validate all MAT cells and convert column-major kernels to FP32."""
    names = {'weights_conv', 'biases_conv', 'prelu_conv'}
    if {key for key in arrays if not key.startswith('__')} != names:
        raise ValueError('Unexpected FSRCNN MAT variable names.')
    for name in names:
        cell = arrays[name]
        if cell.shape != (8, 1) or cell.dtype != object:
            raise ValueError(f'Unexpected cell schema: {name}')
    state = {}
    for layer, shape in enumerate(WEIGHT_SHAPES):
        outputs = shape[2] if layer < 7 else 1
        specifications = (
            ('weights_conv', shape, np.float64),
            ('biases_conv', (outputs, 1), np.float32),
            ('prelu_conv', (1, 1) if layer < 7 else (1, 0),
             np.float32 if layer < 7 else np.float64),
        )
        for name, expected_shape, dtype in specifications:
            value = arrays[name][layer, 0]
            if (value.shape != expected_shape or value.dtype != dtype
                    or not np.isfinite(value).all()):
                raise ValueError(f'Invalid FSRCNN cell {name}[{layer+1}].')
        weights = arrays['weights_conv'][layer, 0]
        if layer < 7:
            size = int(np.sqrt(shape[1]))
            kernels = np.array([
                [weights[i, :, o].reshape(size, size, order='F')
                 for i in range(shape[0])] for o in range(outputs)
            ])
            prefix = f'stages.{layer}.'
            state[prefix+'conv.weight'] = torch.tensor(kernels, dtype=torch.float32)
            state[prefix+'conv.bias'] = torch.tensor(
                arrays['biases_conv'][layer, 0].ravel(), dtype=torch.float32)
            state[prefix+'slope'] = torch.tensor(
                arrays['prelu_conv'][layer, 0].ravel(), dtype=torch.float32)
        else:
            kernels = np.stack([
                weights[0, :, i].reshape(9, 9, order='F') for i in range(56)
            ])[:, None]
            state['deconv_weight'] = torch.tensor(kernels, dtype=torch.float32)
            state['deconv_bias'] = torch.tensor(
                arrays['biases_conv'][layer, 0].ravel(), dtype=torch.float32)
    return state


def shared_prelu(image: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    """Apply the MATLAB shared scalar PReLU, including negative slopes."""
    return image.clamp_min(0) + slope*image.clamp_max(0)


class ConvStage(nn.Module):
    """Replicate-padded cross-correlation, bias, then scalar PReLU."""

    def __init__(self, inputs: int, outputs: int, kernel: int, padding: int) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(inputs, outputs, kernel, padding=0)
        self.slope = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Preserve spatial dimensions before the shared activation."""
        if self.padding:
            image = functional.pad(image, (self.padding,)*4, mode='replicate')
        return shared_prelu(self.conv(image), self.slope)


def deconvolve(features: torch.Tensor, weights: torch.Tensor,
               bias: torch.Tensor) -> torch.Tensor:
    """Apply F-D5 scatter geometry with replicate boundaries and one bias."""
    height, width = features.shape[-2:]
    padded = functional.pad(features, (1, 1, 1, 1), mode='replicate')
    output = functional.conv_transpose2d(
        padded, weights, bias=None, stride=4, padding=0, output_padding=0)
    return output[..., 8:8+4*height, 8:8+4*width] + bias.reshape(1, 1, 1, 1)


class Fsrcnn(nn.Module):
    """Seven activated convolutions and the unactivated ×4 deconvolution."""

    def __init__(self) -> None:
        super().__init__()
        channels = (1, 56, 12, 12, 12, 12, 12, 56)
        kernels = (5, 1, 3, 3, 3, 3, 1)
        paddings = (2, 0, 1, 1, 1, 1, 0)
        self.stages = nn.ModuleList([
            ConvStage(channels[i], channels[i+1], kernels[i], paddings[i])
            for i in range(7)
        ])
        self.deconv_weight = nn.Parameter(torch.zeros(56, 1, 9, 9))
        self.deconv_bias = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Map normalized N1HW luminance to N1(4H)(4W)."""
        for stage in self.stages:
            image = stage(image)
        return deconvolve(image, self.deconv_weight, self.deconv_bias)


def load_fsrcnn(path: Path = CHECKPOINT) -> Fsrcnn:
    """Verify the fixed file before reading MAT cells and strictly loading."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('FSRCNN checkpoint SHA-256 differs from locked file.')
    state = convert_weights(loadmat(path))
    model = Fsrcnn()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class FsrcnnOutput:
    """RGB output, diagnostic Y, normalized float Y and source qualification."""

    rgb: NDArray
    y_uint8: NDArray
    y_float: NDArray
    source_notice: str = SOURCE_NOTICE


def super_resolve(model: nn.Module, lr: NDArray) -> FsrcnnOutput:
    """Apply F1–F8 on CPU, with LR luminance sent directly to the network."""
    if (not isinstance(lr, np.ndarray) or lr.dtype != np.uint8 or lr.ndim != 3
            or lr.shape[-1] != 3 or min(lr.shape[:2]) < 1):
        raise ValueError('Expected nonempty LR RGB uint8.')
    ycc = rgb2ycbcr_uint8(lr)
    tensor = torch.from_numpy(ycc[..., 0].astype(np.float32)/255)[None, None]
    with torch.inference_mode():
        y_float = model(tensor)[0, 0].cpu().numpy().copy()
    if not np.isfinite(y_float).all():
        raise ValueError('FSRCNN pre-quantization Y must be finite.')
    y_uint8 = matlab_uint8_round(y_float.astype(np.float64)*255)
    channels = [y_uint8]
    for channel in (1, 2):
        channels.append(matlab_uint8_round(imresize_bicubic_float(
            ycc[..., channel].astype(np.float64), 4)))
    rgb = ycbcr2rgb_uint8(np.stack(channels, axis=-1))
    return FsrcnnOutput(rgb, y_uint8, y_float)
