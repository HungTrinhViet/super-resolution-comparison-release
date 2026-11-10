"""Synthetic-only input, metric and exclusive demo-export checks."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from sr_compare.gui import controller
from sr_compare.metrics import evaluate


def prediction(rgb):
    return SimpleNamespace(rgb=rgb, algorithm='Bicubic',
                           checkpoint_sha256=None, source_notice='synthetic')


@pytest.mark.parametrize('mode', ['L', 'RGBA', 'P'])
def test_mode_conversion_is_explicit_and_pixel_exact(tmp_path, mode):
    path = tmp_path / 'input.png'
    original = Image.new(mode, (7, 5))
    original.save(path)
    loaded = controller.load_input(path)
    assert loaded.original_mode == mode
    assert mode in loaded.conversion_notice
    assert 'RGB' in loaded.conversion_notice
    np.testing.assert_array_equal(loaded.rgb, np.array(original.convert('RGB')))
    assert loaded.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('suffix', ['png', 'jpg', 'bmp'])
def test_rgb_formats_have_no_conversion_notice(tmp_path, suffix):
    path = tmp_path / ('input.' + suffix)
    Image.new('RGB', (7, 5)).save(path)
    loaded = controller.load_input(path)
    assert loaded.conversion_notice is None
    assert loaded.rgb.shape == (5, 7, 3)


@pytest.mark.parametrize('size,expected', [((256, 256), False),
                                          ((257, 2), True), ((2, 257), True)])
def test_large_image_warning_requires_confirmation(tmp_path, size, expected):
    path = tmp_path / 'input.png'
    Image.new('RGB', size).save(path)
    loaded = controller.load_input(path)
    assert loaded.requires_confirmation is expected
    assert bool(loaded.size_warning) is expected


def test_exif_orientation_is_not_silently_changed(tmp_path):
    path = tmp_path / 'input.jpg'
    image = Image.new('RGB', (7, 5))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    with pytest.raises(ValueError, match='Ảnh có hướng EXIF khác chuẩn. Hãy xoay ảnh trước khi mở.'):
        controller.load_input(path)


def test_other_formats_are_rejected(tmp_path):
    path = tmp_path / 'input.gif'
    Image.new('RGB', (7, 5)).save(path)
    with pytest.raises(ValueError, match='PNG|JPG|BMP'):
        controller.load_input(path)


def test_invalid_hr_is_not_evaluated_or_resized(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid HR must never enter M1 evaluation')
    monkeypatch.setattr(controller, 'evaluate', forbidden)
    lr = np.zeros((6, 7, 3), dtype=np.uint8)
    sr = np.zeros((24, 28, 3), dtype=np.uint8)
    wrong_hr = np.zeros((25, 28, 3), dtype=np.uint8)
    result = controller.process_result(lr, prediction(sr), .25, hr=wrong_hr)
    assert result.metrics is None
    assert 'HR' in result.metric_message
    assert wrong_hr.shape == (25, 28, 3)
    assert controller.process_result(lr, prediction(sr), .2).metrics is None


def test_valid_hr_reuses_m1_unchanged_and_labels_demo():
    lr = np.zeros((6, 7, 3), dtype=np.uint8)
    sr = np.full((24, 28, 3), 10, dtype=np.uint8)
    hr = np.full_like(sr, 20)
    result = controller.process_result(lr, prediction(sr), .25, hr=hr)
    assert result.metrics == evaluate(hr, sr)
    assert result.elapsed_seconds == .25
    assert result.metric_label == 'Theo giao thức M1; không phải kết quả benchmark'
    assert result.time_label == 'Thời gian demo trong GUI, không phải kết quả benchmark đã nghiệm thu.'
    assert controller.hr_caveat() == ('PSNR/SSIM chỉ có ý nghĩa khi HR là ảnh gốc đúng của LR '
                                     'và được căn chỉnh khớp. Chương trình chỉ kiểm được kích thước.')
    assert controller.hr_caveat() in result.message


def test_too_small_hr_reports_metric_limit_without_losing_sr(caplog):
    lr = np.zeros((2, 2, 3), dtype=np.uint8)
    sr = np.zeros((8, 8, 3), dtype=np.uint8)
    result = controller.process_result(lr, prediction(sr), .1, hr=sr.copy())
    assert result.metrics is None
    assert '11x11' in result.metric_message
    assert 'Metrics need' not in result.metric_message
    assert 'Metrics need' in caplog.text
    np.testing.assert_array_equal(result.rgb, sr)


def test_save_is_exclusive_and_preserves_provenance(tmp_path):
    path = tmp_path / 'input.png'
    Image.new('RGB', (6, 7)).save(path)
    loaded = controller.load_input(path)
    sr = np.zeros((28, 24, 3), dtype=np.uint8)
    result = controller.process_result(loaded.rgb, prediction(sr), .5, hr=sr)
    output = tmp_path / 'new-output'
    png, sidecar = controller.save_result(result, loaded, output)
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    assert metadata['input_file_sha256'] == loaded.file_sha256
    assert metadata['input_rgb_sha256'] == hashlib.sha256(loaded.rgb.tobytes()).hexdigest()
    assert metadata['output_rgb_sha256'] == hashlib.sha256(sr.tobytes()).hexdigest()
    assert metadata['output_file_sha256'] == hashlib.sha256(png.read_bytes()).hexdigest()
    assert metadata['time_label'] == controller.DEMO_TIME_LABEL
    assert metadata['metric_label'] == controller.METRIC_LABEL
    assert metadata['metrics']['y']['psnr'] == 'Infinity'
    assert metadata['source_notice'] == 'synthetic'
    assert metadata['checkpoint_sha256'] is None
    assert {'python', 'numpy', 'Pillow', 'torch', 'scikit-image'} <= metadata['versions'].keys()
    saved = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        controller.save_result(result, loaded, output)
    assert saved == {p.name: p.read_bytes() for p in output.iterdir()}


def test_save_rejects_a_different_input_before_creating_output(tmp_path):
    path = tmp_path / 'input.png'
    Image.new('RGB', (6, 7), 'white').save(path)
    loaded = controller.load_input(path)
    lr = np.zeros_like(loaded.rgb)
    result = controller.process_result(lr, prediction(np.zeros((28, 24, 3), dtype=np.uint8)), .1)
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='Ảnh đầu vào'):
        controller.save_result(result, loaded, output)
    assert not output.exists()


def test_metric_error_details_stay_in_logs(monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise ValueError('synthetic technical metric failure')
    monkeypatch.setattr(controller, 'evaluate', fail)
    lr = np.zeros((6, 7, 3), dtype=np.uint8)
    sr = np.zeros((24, 28, 3), dtype=np.uint8)
    result = controller.process_result(lr, prediction(sr), .1, hr=sr)
    assert result.metrics is None
    assert 'synthetic technical metric failure' not in result.metric_message
    assert 'synthetic technical metric failure' in caplog.text
    assert 'Không tính được PSNR/SSIM' in result.metric_message


def test_gui_strings_do_not_use_long_dashes():
    forbidden = []
    for path in Path(controller.__file__).parent.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        forbidden.extend((path.name, node.lineno) for node in ast.walk(tree)
                         if isinstance(node, ast.Constant) and isinstance(node.value, str)
                         and ('\u2014' in node.value or '\u2013' in node.value))
    assert forbidden == []


def test_wrong_output_size_is_short_vietnamese():
    lr = np.zeros((6, 7, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='Kết quả phải có kích thước gấp 4 lần ảnh LR.'):
        controller.process_result(lr, prediction(lr), .1)
