"""Synthetic exact-integer tests for exploratory low-gradient, mid-luminance crops."""
import numpy as np
import pytest

from sr_compare.comparison_crops import integer_y
from sr_compare.comparison_crops_v2 import select_lowgrad_mid, annotate_v1


def grey_for_y(target):
    values = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)
    return int(np.flatnonzero(integer_y(values).ravel() == target)[0])


def boundary_image(mean):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :32] = grey_for_y(mean-2)
    image[:, 32:] = grey_for_y(mean+2)
    return image


@pytest.mark.parametrize('mean', [60, 200])
def test_exact_mean_and_std_boundaries(mean):
    record = select_lowgrad_mid(boundary_image(mean))
    assert record['sum_y'] == mean*4096
    assert record['variance_numerator'] == 4*4096**2
    assert record['mean_y'] == mean
    assert record['exploratory'] is True
    assert (record['x'], record['y']) == (0, 0)


@pytest.mark.parametrize('mean', [59, 201])
def test_mean_outside_interval_rejected(mean):
    with pytest.raises(ValueError, match='eligible'):
        select_lowgrad_mid(boundary_image(mean))


@pytest.mark.parametrize('mean,delta', [(60, -1), (200, 1)])
def test_mean_one_integer_sum_unit_outside_rejected(mean, delta):
    image = np.full((64, 64, 3), grey_for_y(mean-3), dtype=np.uint8)
    image[:, 32:] = grey_for_y(mean+3)
    image[0, 0] = grey_for_y(mean-3+delta)
    y = integer_y(image)
    assert int(y.sum()) == mean*4096+delta
    assert 4096*int((y*y).sum())-int(y.sum())**2 > 4*4096**2
    with pytest.raises(ValueError, match='eligible'):
        select_lowgrad_mid(image)


def test_std_tie_invalid_and_duplicate():
    with pytest.raises(ValueError, match='eligible'):
        select_lowgrad_mid(np.full((70, 75, 3), 100, dtype=np.uint8))
    with pytest.raises(ValueError):
        select_lowgrad_mid(np.zeros((63, 65, 3), dtype=np.uint8))
    image = np.full((65, 65, 3), grey_for_y(58), dtype=np.uint8)
    image[1::2] = grey_for_y(62)
    record = select_lowgrad_mid(image)
    assert (record['x'], record['y']) == (0, 0)
    annotated = annotate_v1(record, [{'x': 0, 'y': 0, 'rule': 'hr_lowgrad64_v1'}])
    assert annotated['matches_v1'] == ['hr_lowgrad64_v1']
    assert annotated['duplicate_coordinates_with_v1'] is True
    assert (annotated['x'], annotated['y']) == (0, 0)
    assert 'matches_v1' not in record


def test_min_gradient_matches_bruteforce():
    image = np.random.default_rng(29).integers(60, 200, (66, 68, 3), dtype=np.uint8)
    y = integer_y(image)
    candidates = []
    for row in range(3):
        for column in range(5):
            window = y[row:row+64, column:column+64]
            total, total2 = int(window.sum()), int((window*window).sum())
            variance = 4096*total2-total*total
            if variance >= 4*4096**2 and 60*4096 <= total <= 200*4096:
                score = int((np.diff(window, axis=0)**2).sum()+(np.diff(window, axis=1)**2).sum())
                candidates.append((score, row, column))
    expected = min(candidates)
    actual = select_lowgrad_mid(image)
    assert (actual['gradient_score'], actual['y'], actual['x']) == expected
