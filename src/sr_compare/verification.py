"""Separate dimension, content and degradation-provenance screening."""
import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .bicubic import quantize, resize


def difference(reference: NDArray, result: NDArray) -> dict:
    """Compare same-grid arrays without crop or implicit resize."""
    if reference.shape != result.shape or reference.size == 0:
        raise ValueError('Comparison requires equal, nonempty shapes.')
    delta = reference.astype(np.float64) - result.astype(np.float64)
    mse = float(np.mean(delta**2))
    return {'mae': float(np.mean(np.abs(delta))),
            'max_abs': float(np.max(np.abs(delta))),
            'psnr': math.inf if mse == 0 else 10 * math.log10(255**2 / mse),
            'exact_match': bool(np.array_equal(reference, result))}


def _passes(row: dict) -> bool:
    return row['psnr'] >= 35 and row['mae'] <= 3


def _screening_view(image: NDArray) -> NDArray:
    """Make a 32x32 content-only view, not a replacement reference image."""
    return np.asarray(Image.fromarray(image).resize(
        (32, 32), Image.Resampling.BICUBIC))


def screen_content(pairs: list[tuple[str, NDArray, NDArray]]) -> dict:
    """Screen pairs and ordered swaps; expose invalid thresholds."""
    matched, negatives = [], []
    generated = [quantize(resize(hr, .25)) for _, hr, _ in pairs]
    for i, (name, _, lr) in enumerate(pairs):
        row = difference(generated[i], lr)
        normalized = difference(_screening_view(generated[i]),
                                _screening_view(lr))
        matched.append({'name': name, **row, 'screen_pass': _passes(row),
                        'normalized_32x32': normalized})
        for j, (other_name, _, other_lr) in enumerate(pairs):
            if i == j:
                continue
            # Wrong pairs have different sizes: use the same normalized view
            # for all controls, while preserving their dimension failure.
            comparison = difference(_screening_view(generated[i]),
                                    _screening_view(other_lr))
            negatives.append({
                'hr': name, 'wrong_lr': other_name,
                'original_shapes_equal': generated[i].shape == other_lr.shape,
                **comparison, 'screen_pass': _passes(comparison)})
    exceeded = sum(row['screen_pass'] for row in negatives)
    return {'status': 'PASS' if all(r['screen_pass'] for r in matched)
            and exceeded == 0 else 'FAIL',
            'threshold': {'psnr_min': 35, 'mae_max': 3, 'data_range': 255},
            'matched': matched, 'negative_controls': negatives,
            'negative_exceed_count': exceeded,
            'control_protocol': (
                '32x32 Pillow BICUBIC content-only views for both matched '
                'and swapped comparisons; original benchmark pairs '
                'remain unchanged.')}
