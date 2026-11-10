"""Author SRCNN 9-5-5 weights, replicate convolutions and RGB adapter P2–P8."""
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

CHECKPOINT_SHA256 = '3f94150d24e5b60dab2b74f4eaf358aefdc72cbe724d86990435bf877ed452ed'
SHAPES = {'weights_conv1': (81, 64), 'biases_conv1': (64, 1),
          'weights_conv2': (64, 25, 32), 'biases_conv2': (32, 1),
          'weights_conv3': (32, 25), 'biases_conv3': (1, 1)}


def convert_weights(arrays: dict[str, NDArray]) -> dict[str, torch.Tensor]:
    """Convert validated author column-major kernels to FP32 OIHW tensors."""
    visible = {key for key in arrays if not key.startswith('__')}
    if visible != set(SHAPES):
        raise ValueError('Unexpected MAT variable names.')
    for name, shape in SHAPES.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f'Invalid MAT shape or nonfinite values: {name}')
    first = np.stack([arrays['weights_conv1'][:, o].reshape(9, 9, order='F')
                      for o in range(64)])[:, None]
    second = np.array([[arrays['weights_conv2'][i, :, o].reshape(5, 5, order='F')
                        for i in range(64)] for o in range(32)])
    third = np.stack([arrays['weights_conv3'][i].reshape(5, 5, order='F')
                      for i in range(32)])[None]
    state = {}
    for layer, weights in enumerate((first, second, third), 1):
        state[f'conv{layer}.weight'] = torch.tensor(weights, dtype=torch.float32)
        state[f'conv{layer}.bias'] = torch.tensor(
            arrays[f'biases_conv{layer}'].ravel(), dtype=torch.float32)
    return state


class Srcnn(nn.Module):
    """Three cross-correlations with replicate boundaries and two ReLUs."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 9)
        self.conv2 = nn.Conv2d(64, 32, 5)
        self.conv3 = nn.Conv2d(32, 1, 5)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Preserve NCHW spatial size using explicit replicate padding."""
        image = functional.relu(self.conv1(functional.pad(image, (4, 4, 4, 4),
                                                          mode='replicate')))
        image = functional.relu(self.conv2(functional.pad(image, (2, 2, 2, 2),
                                                          mode='replicate')))
        return self.conv3(functional.pad(image, (2, 2, 2, 2), mode='replicate'))


def load_srcnn(path: Path) -> Srcnn:
    """Check the fixed MAT bytes/schema before strictly loading CPU weights."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('SRCNN checkpoint SHA-256 differs from verified M2 file.')
    arrays = loadmat(path)
    if any(arrays[name].dtype != np.float64 for name in SHAPES if name in arrays):
        raise ValueError('Author MAT dtype differs from verified float64.')
    model = Srcnn()
    model.load_state_dict(convert_weights(arrays), strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class SrcnnOutput:
    """RGB result and P6 diagnostic Y with pre-quantization normalized Y."""

    rgb: NDArray
    y_uint8: NDArray
    y_float: NDArray


def super_resolve(model: Srcnn, lr: NDArray) -> SrcnnOutput:
    """Apply locked P2–P8 to LR RGB uint8 on CPU, without reference HR."""
    if (not isinstance(lr, np.ndarray) or lr.dtype != np.uint8 or lr.ndim != 3
            or lr.shape[-1] != 3 or min(lr.shape[:2]) < 1):
        raise ValueError('Expected nonempty LR RGB uint8.')
    ycc = rgb2ycbcr_uint8(lr)
    enlarged = imresize_bicubic_float(ycc[..., 0].astype(np.float64)/255, 4)
    tensor = torch.from_numpy(enlarged.astype(np.float32))[None, None]
    with torch.inference_mode():
        y_float = model(tensor)[0, 0].cpu().numpy().copy()
    if not np.isfinite(y_float).all():
        raise ValueError('SRCNN pre-quantization Y must be finite.')
    y_uint8 = matlab_uint8_round(y_float.astype(np.float64)*255)
    channels = [y_uint8]
    for channel in (1, 2):
        channels.append(matlab_uint8_round(imresize_bicubic_float(
            ycc[..., channel].astype(np.float64), 4)))
    rgb = ycbcr2rgb_uint8(np.stack(channels, axis=-1))
    return SrcnnOutput(rgb, y_uint8, y_float)
