"""Exact integer HR-only crop rules and locked deterministic image rendering."""
import numpy as np
import pytest

from sr_compare.comparison_crops import select_crops, window_sums, integer_y


def test_selectors_match_all_bruteforce_windows():
    hr = np.random.default_rng(8).integers(0, 256, (66, 68, 3), dtype=np.uint8)
    y = integer_y(hr)
    values = []
    for row in range(3):
        for column in range(5):
            window = y[row:row+64, column:column+64]
            score = int((np.diff(window, axis=0)**2).sum()
                        +(np.diff(window, axis=1)**2).sum())
            numerator = 4096*int((window*window).sum())-int(window.sum())**2
            values.append((score, row, column, numerator))
    high = min(values, key=lambda v: (-v[0], v[1], v[2]))
    low = min((v for v in values if v[3] >= 4*4096**2), key=lambda v: v[:3])
    crops = select_crops(hr)
    for record, expected in zip(crops[:2], (high, low)):
        assert (record['gradient_score'], record['y'], record['x'],
                record['variance_numerator']) == expected


def test_render_panel_exact_crop_pixels():
    from sr_compare.comparison_crops import render_panel
    hr = np.random.default_rng(44).integers(0, 256, (70, 73, 3), dtype=np.uint8)
    original = hr.copy()
    crop = {'rule': 'unit_only', 'x': 3, 'y': 5}
    panel = render_panel([(label, 'synthetic_only', hr) for label in
                          ('HR', 'Bicubic', 'SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN')], 'unit', crop)
    expected = np.repeat(np.repeat(hr[5:69, 3:67], 4, 0), 4, 1)
    for column in range(6):
        x = 8+column*264
        np.testing.assert_array_equal(np.array(panel)[132:388, x:x+256], expected)
    np.testing.assert_array_equal(hr, original)


def test_lock_verifies_before_any_sr_read(tmp_path, monkeypatch):
    import json
    from PIL import Image
    import sr_compare.comparison_crops as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    for name in module.CODE:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic selector code only')
    hr_dir = tmp_path/'data'
    hr_dir.mkdir()
    hr = np.zeros((64, 64, 3), dtype=np.uint8)
    hr[:, 1::2] = 255
    for index in range(1, 6):
        Image.fromarray(hr).save(hr_dir/f'img_{index:03d}_SRF_4_HR.png')
    manifest, receipt = tmp_path/'manifest.json', tmp_path/'receipt.json'
    module.lock_manifest(manifest, receipt, hr_dir)
    data = module.verify_manifest(manifest, receipt)
    assert len(data['records']) == 15
    assert all(r['duplicate_coordinates'] for r in data['records'])
    with pytest.raises(FileExistsError):
        module.lock_manifest(manifest, receipt, hr_dir)
    changed = json.loads(manifest.read_text())
    changed['records'][0]['x'] = 1
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='manifest hash'):
        module.verify_manifest(manifest, receipt)


def test_std_boundary_tie_and_duplicate():
    hr = np.zeros((64, 64, 3), dtype=np.uint8)
    hr[:, 1::2] = 255
    crops = select_crops(hr)
    assert len(crops) == 3
    assert all(c['x'] == c['y'] == 0 and c['duplicate_coordinates'] for c in crops)
    assert all(c['sum_y'] == int(integer_y(hr).sum()) for c in crops)
    low = next(c for c in crops if c['rule'] == 'hr_lowgrad64_v1')
    assert low['variance_numerator'] >= 4*4096**2


def test_known_gradient_location_and_tie():
    hr = np.zeros((64, 128, 3), dtype=np.uint8)
    hr[:, 64:] = np.repeat(((np.arange(64) % 2)*255)[None, :, None], 64, axis=0)
    crops = select_crops(hr)
    high = next(c for c in crops if c['rule'] == 'hr_gradient64_v1')
    assert high['x'] == 64 and high['y'] == 0
    assert next(c for c in crops if c['rule'] == 'center64_v1')['x'] == 32


def test_population_std_exact_integer_boundary():
    for level, qualifies in ((4, False), (5, True)):
        hr = np.zeros((64, 64, 3), dtype=np.uint8)
        hr[:, 32:] = level
        if qualifies:
            crops = select_crops(hr)
            assert len(crops) == 3
            assert crops[0]['variance_numerator'] == 4*4096**2
        else:
            with pytest.raises(ValueError, match='eligible'):
                select_crops(hr)


def test_invalid_and_no_eligible_window():
    with pytest.raises(ValueError):
        select_crops(np.zeros((63, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='eligible'):
        select_crops(np.zeros((70, 75, 3), dtype=np.uint8))
    values = np.arange(80).reshape(8, 10).astype(np.int64)
    expected = np.array([[values[y:y+3, x:x+4].sum() for x in range(7)] for y in range(6)])
    np.testing.assert_array_equal(window_sums(values, 3, 4), expected)
