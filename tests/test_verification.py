"""Dataset screening tests use synthetic images only."""
import numpy as np
import pytest

from sr_compare.verification import difference, screen_content


def test_difference_exact_and_shape():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = difference(image, image)
    assert result['exact_match']
    assert result['max_abs'] == 0
    assert np.isposinf(result['psnr'])
    with pytest.raises(ValueError):
        difference(image, image[:4])


def test_negative_control_rejects_non_discriminating_threshold():
    image = np.full((24, 24, 3), 80, dtype=np.uint8)
    lr = np.full((6, 6, 3), 80, dtype=np.uint8)
    result = screen_content([('first', image, lr), ('second', image, lr)])
    assert result['negative_exceed_count'] == 2
    assert result['status'] == 'FAIL'
