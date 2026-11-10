"""T6: float interpolation against independently applied contributions."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sr_compare.matlab_resize import imresize_bicubic_float


def independent_resize(image):
    path = Path(__file__).parents[1] / 'scripts/crosscheck_bicubic.py'
    spec = importlib.util.spec_from_file_location('float_reference', path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    result = image.copy()
    for axis in [0, 1]:
        weights, indices = checker.contributions(result.shape[axis])
        source = np.moveaxis(result, axis, 0)
        output = np.empty((len(weights), *source.shape[1:]))
        for pixel in range(len(weights)):
            output[pixel] = np.sum(weights[pixel, :, None]
                                   * source[indices[pixel]], axis=0)
        result = np.moveaxis(output, 0, axis)
    return result


@pytest.mark.parametrize('shape', [(1, 1), (1, 2), (2, 1), (3, 5), (7, 4)])
def test_t6a_independent_float_and_boundaries(shape):
    image = np.random.default_rng(42).random(shape)
    actual = imresize_bicubic_float(image, 4)
    error = np.max(np.abs(actual - independent_resize(image)))
    print('T6a', shape, 'max error', error)
    assert actual.shape == (shape[0]*4, shape[1]*4)
    assert actual.dtype == np.float64
    assert error <= 1e-9
    np.testing.assert_allclose(imresize_bicubic_float(np.full(shape, .37), 4),
                               .37, rtol=0, atol=1e-9)


def test_t6b_linearity_and_normalization():
    rng = np.random.default_rng(37)
    x, y = rng.random((2, 5, 7))
    resize = lambda a: imresize_bicubic_float(a, 4)
    errors = [np.max(np.abs(resize(.3*x + .7*y) - .3*resize(x) - .7*resize(y))),
              np.max(np.abs(resize(x/255) - resize(x)/255))]
    float_error = np.max(np.abs(resize(x).astype(np.float32) - resize(x)))
    print('T6b float64 errors', errors, 'float32 error', float_error)
    assert max(errors) <= 1e-12
    assert float_error <= 1e-6


def test_resize_invalid_and_no_quantization():
    for bad in [np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 3)),
                np.empty((0, 2)), np.array([[np.inf]])]:
        with pytest.raises(ValueError):
            imresize_bicubic_float(bad, 4)
    with pytest.raises(ValueError):
        imresize_bicubic_float(np.ones((2, 2)), .25)
    image = np.array([[0., 0., 1., 1.]])
    actual = imresize_bicubic_float(image, 4)
    assert actual.min() < 0 and actual.max() > 1
