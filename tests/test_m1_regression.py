"""Compare M1 forward outputs with previously accepted artifacts, no timing."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import check_set5
from sr_compare.bicubic import upscale
from sr_compare.io import load_pair
from sr_compare.metrics import evaluate

ABSOLUTE_TOLERANCE = 1e-9
EXPECTED_PATH = Path(__file__).parent / 'fixtures/m1_bicubic_expected.json'


@pytest.mark.parametrize('image_index', range(5))
def test_m1_regression(set5_dir: Path, image_index: int) -> None:
    """Assert the serialized PNG hash and all six metrics remain unchanged."""
    golden = json.loads(EXPECTED_PATH.read_text(encoding='utf-8'))
    assert golden['source_run_id'] == 'set5_numpy_x4_20261108_1627'
    assert golden['source_commit'] == '100c90b'
    expected = golden['images'][image_index]
    name = expected['name']
    hr, lr = load_pair(set5_dir / f'{name}_HR.png',
                       set5_dir / f'{name}_LR.png')
    result = upscale(lr)
    with BytesIO() as stream:
        Image.fromarray(result).save(stream, format='PNG')
        actual_hash = hashlib.sha256(stream.getvalue()).hexdigest()
    assert actual_hash == expected['sr_png_sha256']
    actual = evaluate(hr, result)
    for space in ['y', 'rgb']:
        for metric in ['mse', 'psnr', 'ssim']:
            assert actual[space][metric] == pytest.approx(
                expected[space][metric], rel=0, abs=ABSOLUTE_TOLERANCE)


def test_missing_set5_skips_by_default(tmp_path: Path) -> None:
    """Missing synthetic paths must be an explicit unverified skip."""
    with pytest.raises(pytest.skip.Exception, match='NOT VERIFIED: Set5 missing'):
        check_set5(tmp_path / 'absent', required=False)


def test_missing_set5_fails_acceptance(tmp_path: Path) -> None:
    """Required real-data acceptance must fail when synthetic paths are absent."""
    with pytest.raises(pytest.fail.Exception, match='NOT VERIFIED: Set5 missing'):
        check_set5(tmp_path / 'absent', required=True)
