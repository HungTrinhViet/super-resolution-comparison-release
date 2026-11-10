"""Locked T7 color and quantization checks, independent expected values."""
import itertools

import numpy as np
import pytest

from sr_compare.matlab_color import (
    matlab_uint8_round, rgb2ycbcr_float, rgb2ycbcr_uint8,
    ycbcr2rgb_float, ycbcr2rgb_uint8,
)


def test_t7a_float_roundtrip():
    rgb = np.array(list(itertools.product(range(0, 256, 17), repeat=3)),
                   dtype=np.float64)
    error = np.max(np.abs(ycbcr2rgb_float(rgb2ycbcr_float(rgb)) - rgb))
    print('T7a max error:', error)
    assert error <= 1e-9


def test_t7b_known_rgb():
    rgb = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                    [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    expected = [[16, 128, 128], [235, 128, 128], [81, 90, 240],
                [145, 54, 34], [41, 240, 110]]
    np.testing.assert_array_equal(rgb2ycbcr_uint8(rgb), expected)


def test_t7c_locked_grid():
    rgb = np.array(list(itertools.product(range(0, 256, 5), repeat=3)),
                   dtype=np.uint8)
    error = np.abs(ycbcr2rgb_uint8(rgb2ycbcr_uint8(rgb)).astype(int)
                   - rgb.astype(int))
    values, counts = np.unique(error, return_counts=True)
    print('T7c absolute channel-error distribution:', dict(zip(values, counts)))
    assert error.max() <= 2


def test_t7d_independent_inverse_coefficients():
    ycc = np.array([[16, 128, 128], [235, 128, 128], [126, 128, 128],
                    [81, 90, 240], [145, 54, 34], [41, 240, 110],
                    [255, 128, 255], [16, 16, 16]], dtype=np.uint8)
    y, cb, cr = (ycc.astype(float) - [16, 128, 128]).T
    independent = np.stack([1.164383*y + 1.596027*cr,
                            1.164383*y - .391762*cb - .812968*cr,
                            1.164383*y + 2.017232*cb], axis=-1)
    expected = [[0, 0, 0], [255, 255, 255], [128, 128, 128],
                [254, 0, 0], [0, 255, 1], [0, 0, 255],
                [255, 175, 255], [0, 135, 0]]
    np.testing.assert_array_equal(np.clip(np.floor(independent + .5), 0, 255),
                                  expected)
    np.testing.assert_array_equal(ycbcr2rgb_uint8(ycc), expected)


def test_t7e_round_saturation_and_nonfinite():
    values = [-.6, -.5, -.49, .49, .5, 1.5, 2.5, 254.5, 255.4, 255.5, 300]
    np.testing.assert_array_equal(matlab_uint8_round(np.array(values)),
                                  [0, 0, 0, 0, 1, 2, 3, 255, 255, 255, 255])
    for value in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            matlab_uint8_round(np.array([value]))
