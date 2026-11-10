"""Test the independent checker without importing the production package."""
import importlib.util
from pathlib import Path

import numpy as np


def test_independent_kernel_and_constant_image():
    path = Path(__file__).parents[1] / 'scripts' / 'crosscheck_bicubic.py'
    spec = importlib.util.spec_from_file_location('independent_checker', path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    np.testing.assert_allclose(
        checker.cubic(np.array([0, .5, 1, 1.5, 2])),
        [1, .5625, 0, -.0625, 0])
    result = checker.upsample(np.full((5, 7, 3), 73, dtype=np.uint8))
    assert result.shape == (20, 28, 3)
    np.testing.assert_array_equal(result, 73)
