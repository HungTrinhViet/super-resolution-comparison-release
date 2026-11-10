"""Locked BasicSR EDSR-baseline x4 and RGB half-even CPU adapter."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

CHECKPOINT = Path('data/checkpoints/edsr/EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth')
CHECKPOINT_SHA256 = '0c287733e70d1b8b8fc5885613ecbe451e5f3010bcae0307612ef5e4aa08dd5f'
RGB_MEAN = (.4488, .4371, .4040)


def subtract_mean(image: torch.Tensor) -> torch.Tensor:
    """Apply BasicSR RGB [0,1] to centered 0..255 intensities."""
    mean = image.new_tensor(RGB_MEAN).view(1, 3, 1, 1)
    return (image - mean)*255


def add_mean(image: torch.Tensor) -> torch.Tensor:
    """Return centered network intensities to unbounded RGB [0,1] scale."""
    mean = image.new_tensor(RGB_MEAN).view(1, 3, 1, 1)
    return image/255 + mean


class ResidualBlock(nn.Module):
    """Two zero-padded convolutions with ReLU and unit residual scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.res_scale = 1
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Keep the identity outside the two-convolution residual branch."""
        return image + self.conv2(self.relu(self.conv1(image)))*self.res_scale


class Edsr(nn.Module):
    """16-block/64-channel baseline with exact BasicSR state dictionary names."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock() for _ in range(16)))
        self.conv_after_body = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(64, 256, 3, padding=1),
                                      nn.PixelShuffle(2))
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run centered RGB features with local/global skips and two shuffles."""
        head = self.conv_first(subtract_mean(image))
        residual = self.conv_after_body(self.body(head)) + head
        return add_mean(self.conv_last(self.upsample(residual)))


def load_edsr(path: Path = CHECKPOINT) -> Edsr:
    """Verify bytes, require params/FP32 tensors and load strictly on CPU."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('EDSR checkpoint SHA-256 differs from verified M2 bytes.')
    wrapper = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(wrapper, dict) or set(wrapper) != {'params'}:
        raise ValueError('EDSR checkpoint must have exactly the params wrapper.')
    state = wrapper['params']
    if not isinstance(state, dict) or len(state) != 74:
        raise ValueError('EDSR requires exactly 74 state entries.')
    for value in state.values():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError('EDSR weights must be finite FP32 tensors.')
    model = Edsr()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 1517571:
        raise ValueError('EDSR parameter count differs from locked baseline.')
    return model.eval()


def prepare_input(lr: NDArray) -> torch.Tensor:
    """Preserve RGB order while converting uint8 HWC to FP32 NCHW [0,1]."""
    if (not isinstance(lr, np.ndarray) or lr.dtype != np.uint8 or lr.ndim != 3
            or lr.shape[-1] != 3 or min(lr.shape[:2]) < 1):
        raise ValueError('Expected nonempty RGB uint8 input.')
    chw = np.ascontiguousarray(lr.transpose(2, 0, 1))
    return torch.from_numpy(chw).to(dtype=torch.float32).unsqueeze(0)/255


def quantize_values(values: torch.Tensor) -> torch.Tensor:
    """Reject nonfinite values, clamp 0..255 and round ties to even."""
    if not torch.isfinite(values).all():
        raise ValueError('EDSR pre-quantization values must be finite.')
    return torch.round(values.clamp(0, 255)).to(torch.uint8)


def tensor_to_rgb(values: torch.Tensor) -> NDArray:
    """Convert unquantized CHW pixel intensities to RGB uint8 HWC."""
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError('Expected a three-channel CHW tensor.')
    return quantize_values(values).permute(1, 2, 0).cpu().numpy().copy()


@dataclass(frozen=True)
class EdsrOutput:
    """RGB pixels plus unclipped HWC float intensities and exact-half counts."""

    rgb: NDArray
    prequant_float: NDArray
    half_tie_count: int
    half_tie_channel_count: int


def super_resolve(model: Edsr, lr: NDArray) -> EdsrOutput:
    """Apply locked E2–E4 on CPU without reference HR, chop or ensemble."""
    with torch.inference_mode():
        intensities = model(prepare_input(lr))[0]*255
        if not torch.isfinite(intensities).all():
            raise ValueError('EDSR pre-quantization values must be finite.')
        ties = (intensities - torch.floor(intensities)) == .5
        result = EdsrOutput(
            tensor_to_rgb(intensities),
            intensities.permute(1, 2, 0).cpu().numpy().copy(),
            int(ties.any(dim=0).sum()), int(ties.sum()))
    return result
