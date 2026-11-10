"""Original ESRGAN architecture; BasicSR converted checkpoint chain UNVERIFIED."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as functional

CHECKPOINT = Path('data/checkpoints/esrgan/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth')
CHECKPOINT_SHA256 = 'ff704c30ff560305e48ed1f4db895e525ab6bc81a46fafe80c0094a271c806d9'
SOURCE_NOTICE = ('Original ESRGAN architecture; BasicSR converted checkpoint; '
                 'conversion chain from original author weights UNVERIFIED; '
                 'weight license and training overlap with Set5 UNVERIFIED.')


class Rdb(nn.Module):
    """Five dense cross-correlations with growth32 and residual scaling0.2."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(160, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 64, 3, padding=1)
        self.lrelu = nn.LeakyReLU(.2, inplace=True)
        self.res_scale = .2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Preserve original-input-first dense concatenation and final linear conv."""
        first = self.lrelu(self.conv1(image))
        second = self.lrelu(self.conv2(torch.cat((image, first), 1)))
        third = self.lrelu(self.conv3(torch.cat((image, first, second), 1)))
        fourth = self.lrelu(self.conv4(torch.cat((image, first, second, third), 1)))
        residual = self.conv5(torch.cat((image, first, second, third, fourth), 1))
        return image + self.res_scale*residual


class Rrdb(nn.Module):
    """Three residual dense blocks inside a second independently scaled skip."""

    def __init__(self) -> None:
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = Rdb(), Rdb(), Rdb()
        self.res_scale = .2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Scale the full three-RDB output, including its internal identities."""
        residual = self.rdb3(self.rdb2(self.rdb1(image)))
        return image + self.res_scale*residual


class RrdbNet(nn.Module):
    """23 RRDBs, global skip and two nearest resize-convolution stages."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1)
        self.body = nn.Sequential(*(Rrdb() for _ in range(23)))
        self.conv_body = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_hr = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)
        self.lrelu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Map RGB [0,1] directly to unbounded RGB at four times resolution."""
        head = self.conv_first(image)
        features = head + self.conv_body(self.body(head))
        features = self.lrelu(self.conv_up1(functional.interpolate(
            features, scale_factor=2, mode='nearest')))
        features = self.lrelu(self.conv_up2(functional.interpolate(
            features, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(features)))


def load_esrgan(path: Path = CHECKPOINT) -> RrdbNet:
    """Validate fixed BasicSR bytes and strictly load finite FP32 params on CPU."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('ESRGAN checkpoint SHA-256 differs from locked M2 file.')
    wrapper = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(wrapper, dict) or set(wrapper) != {'params'}:
        raise ValueError('ESRGAN checkpoint requires exactly the params wrapper.')
    state = wrapper['params']
    if not isinstance(state, dict) or len(state) != 702:
        raise ValueError('ESRGAN requires exactly 702 state entries.')
    for value in state.values():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError('ESRGAN weights must be finite FP32 tensors.')
    model = RrdbNet()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 16697987:
        raise ValueError('Original ESRGAN architecture parameter count changed.')
    return model.eval()


def prepare_input(lr: NDArray) -> torch.Tensor:
    """Convert nonempty uint8 RGB HWC to batch-one CPU FP32 NCHW [0,1]."""
    if (not isinstance(lr, np.ndarray) or lr.dtype != np.uint8 or lr.ndim != 3
            or lr.shape[-1] != 3 or min(lr.shape[:2]) < 1):
        raise ValueError('Expected nonempty RGB uint8 input.')
    chw = np.ascontiguousarray(lr.transpose(2, 0, 1))
    return torch.from_numpy(chw).to(dtype=torch.float32).unsqueeze(0)/255


def quantize_values(values: torch.Tensor) -> torch.Tensor:
    """Reject nonfinite 0..255-scale values before clamping and half-even round."""
    if not torch.isfinite(values).all():
        raise ValueError('ESRGAN pre-quantization values must be finite.')
    return torch.round(values.clamp(0, 255)).to(torch.uint8)


@dataclass(frozen=True)
class EsrganOutput:
    """RGB pixels, unclamped HWC intensities and qualified checkpoint provenance."""

    rgb: NDArray
    prequant_float: NDArray
    source_notice: str = SOURCE_NOTICE


def super_resolve(model: RrdbNet, lr: NDArray) -> EsrganOutput:
    """Infer with no mean shift, preserving unclamped RGB float intensities."""
    with torch.inference_mode():
        normalized = model(prepare_input(lr))[0]
        if not torch.isfinite(normalized).all():
            raise ValueError('ESRGAN pre-quantization values must be finite.')
        pixels = quantize_values(normalized.clamp(0, 1)*255).permute(1, 2, 0).cpu().numpy().copy()
        values = (normalized*255).permute(1, 2, 0).cpu().numpy().copy()
    return EsrganOutput(pixels, values)
