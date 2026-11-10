"""Nonvisual x4 inference through the five existing, locked library pipelines."""
from dataclasses import dataclass
import gc
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

from . import bicubic, edsr, esrgan, fsrcnn, srcnn
from .io import validate_rgb


ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ('Bicubic', 'SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN')
# SRCNN's accepted path is defined in scripts/run_srcnn.py, not its library.
_SRCNN_CHECKPOINT = Path('data/checkpoints/srcnn/SRCNN/model/9-5-5(ImageNet)/x4.mat')
_MODELS = {
    'SRCNN': (srcnn, 'load_srcnn', _SRCNN_CHECKPOINT),
    'FSRCNN': (fsrcnn, 'load_fsrcnn', fsrcnn.CHECKPOINT),
    'EDSR': (edsr, 'load_edsr', edsr.CHECKPOINT),
    'ESRGAN': (esrgan, 'load_esrgan', esrgan.CHECKPOINT),
}
_threads_configured = False


def configure_threads() -> None:
    """Call at process startup, before any torch operations or worker threads."""
    global _threads_configured
    if not _threads_configured:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        _threads_configured = True


@dataclass(frozen=True)
class InferenceResult:
    """Unmodified RGB pixels and the accepted model's provenance metadata."""

    rgb: NDArray[np.uint8]
    algorithm: str
    checkpoint_sha256: str | None
    source_notice: str


class InferenceRegistry:
    """Use sequentially from one worker; retain at most one loaded CPU model."""

    def __init__(self) -> None:
        self._algorithm: str | None = None
        self._model = None

    def release(self) -> None:
        """Drop the cached model and collect before loading a different model."""
        self._model = None
        self._algorithm = None
        gc.collect()

    def predict(self, algorithm: str, lr: NDArray) -> InferenceResult:
        """Return the exact library output; never resize or convert the input."""
        if algorithm not in ALGORITHMS:
            raise ValueError(f'Unknown algorithm: {algorithm}.')
        validate_rgb(lr)
        if algorithm == 'Bicubic':
            if self._model is not None:
                self.release()
            return InferenceResult(bicubic.upscale(lr, backend='numpy'),
                                   algorithm, None, '')

        module, loader_name, relative_path = _MODELS[algorithm]
        if self._algorithm != algorithm:
            self.release()
            path = ROOT / relative_path
            if not path.is_file():
                message = (f'{algorithm} checkpoint missing: {path}. '
                           'See scripts/download_models.py and README.md '
                           'for checkpoint preparation.')
                if algorithm == 'FSRCNN':
                    message += (' FSRCNN has no automatic download item: use the '
                                'manual source and SHA-256 instructions in README.md.')
                raise FileNotFoundError(message)
            self._model = getattr(module, loader_name)(path)
            self._algorithm = algorithm
        output = module.super_resolve(self._model, lr)
        return InferenceResult(output.rgb, algorithm, module.CHECKPOINT_SHA256,
                               getattr(module, 'SOURCE_NOTICE', ''))
