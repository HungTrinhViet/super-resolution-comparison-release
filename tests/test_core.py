"""Synthetic fixtures test software behavior, never experimental SR quality."""
import json
import numpy as np
from PIL import Image
import pytest

from sr_compare.bicubic import cubic_kernel, resize, upscale, quantize
from sr_compare.degradation import make_lr, modcrop
from sr_compare.io import load_rgb, load_pair, save_rgb
from sr_compare.metrics import evaluate, rgb_to_y
from sr_compare.timing import measure
from sr_compare.experiment import run_experiment, write_json
from sr_compare.dataset import git_blob_sha1, download_set5


def test_identical_metrics():
    image = np.full((24, 28, 3), 100, dtype=np.uint8)
    for metrics in evaluate(image, image).values():
        assert metrics['mse'] == 0
        assert np.isposinf(metrics['psnr'])
        assert metrics['ssim'] == pytest.approx(1)


def test_hand_calculated_psnr():
    hr = np.zeros((24, 24, 3), dtype=np.uint8)
    sr = np.full_like(hr, 10)
    result = evaluate(hr, sr)
    assert result['rgb']['mse'] == 100
    assert result['rgb']['psnr'] == pytest.approx(20 * np.log10(255 / 10))
    assert result['y']['mse'] == pytest.approx((2190 / 255) ** 2)
    first, second = 16, 16 + 2190 / 255
    c1 = (.01 * 255) ** 2
    expected_ssim = (2 * first * second + c1) / (first ** 2 + second ** 2 + c1)
    assert result['y']['ssim'] == pytest.approx(expected_ssim)


def test_bt601_colors():
    colors = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0],
                        [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    np.testing.assert_allclose(rgb_to_y(colors),
                               [[16, 235, 81.481, 144.553, 40.966]])


def test_crop_four_pixels():
    hr = np.zeros((24, 24, 3), dtype=np.uint8)
    sr = hr.copy()
    sr[:4] = sr[-4:] = 255
    sr[:, :4] = sr[:, -4:] = 255
    assert evaluate(hr, sr)['y']['mse'] == 0
    assert evaluate(hr, sr, border=0)['y']['mse'] > 0


@pytest.mark.parametrize('shape', [(24, 23, 3), (18, 18, 3)])
def test_metric_shape_and_small_image(shape):
    hr = np.zeros((24, 24, 3), dtype=np.uint8)
    other = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        evaluate(hr if shape[0] == 24 else other, other)


@pytest.mark.parametrize('bad', [np.zeros((0, 4, 3), dtype=np.uint8),
                               np.zeros((5, 5)),
                               np.full((5, 5, 3), np.nan)])
def test_invalid_images(bad):
    with pytest.raises((ValueError, TypeError)):
        upscale(bad)


@pytest.mark.parametrize('backend', ['numpy', 'pillow'])
def test_bicubic_constant_dimensions_symmetry(backend):
    constant = np.full((5, 7, 3), 73, dtype=np.uint8)
    result = upscale(constant, backend=backend)
    assert result.shape == (20, 28, 3)
    np.testing.assert_array_equal(result, 73)
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    image[2, 3] = 255
    result = upscale(image, backend=backend)
    np.testing.assert_array_equal(result, result[::-1, ::-1])


def test_kernel_and_half_pixel_alignment():
    np.testing.assert_allclose(cubic_kernel(np.array([0, .5, 1, 1.5, 2])),
                               [1, .5625, 0, -.0625, 0])
    ramp = np.broadcast_to(np.arange(8)[:, None, None] * 20,
                           (8, 8, 3)).astype(np.uint8)
    enlarged = resize(ramp, 4)
    # Center-aligned cubic reproduces an interior affine ramp.
    assert enlarged[12, 12, 0] == pytest.approx(((12 + .5) / 4 - .5) * 20)
    assert enlarged[0, 0, 0] == pytest.approx(-2.34375)
    np.testing.assert_allclose(resize(ramp, 1), ramp)


def test_antialias_and_invalid_backend():
    board = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = np.repeat(board[..., None], 3, axis=2)
    np.testing.assert_allclose(resize(image, .25)[2:-2, 2:-2], 127.5)
    with pytest.raises(ValueError):
        upscale(image, backend='unknown')
    with pytest.raises(ValueError):
        resize(image, 0)


def test_quantization():
    np.testing.assert_array_equal(quantize(np.array([-1, .5, 2.5, 300.])),
                                   [0, 1, 3, 255])
    with pytest.raises(ValueError):
        quantize(np.array([np.nan]))


def test_modcrop_only_single_hr(tmp_path):
    image = np.zeros((25, 27, 3), dtype=np.uint8)
    assert modcrop(image).shape == (24, 24, 3)
    assert make_lr(image).shape == (6, 6, 3)
    hr_path, lr_path = tmp_path / 'hr.png', tmp_path / 'lr.png'
    Image.fromarray(image).save(hr_path)
    Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(lr_path)
    with pytest.raises(ValueError, match='shape'):
        load_pair(hr_path, lr_path)


def test_file_handling(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / 'missing.png')
    corrupt = tmp_path / 'corrupt.png'
    corrupt.write_text('not an image')
    with pytest.raises((ValueError, OSError)):
        load_rgb(corrupt)
    for mode in ['L', 'RGBA', 'P', 'I;16']:
        path = tmp_path / (mode.replace(';', '_') + '.png')
        Image.new(mode, (24, 24)).save(path)
        with pytest.raises(ValueError, match='RGB'):
            load_rgb(path)
    path = tmp_path / 'rgb.png'
    image = np.full((24, 24, 3), 77, dtype=np.uint8)
    save_rgb(path, image)
    np.testing.assert_array_equal(load_rgb(path), image)
    with pytest.raises(FileExistsError):
        save_rgb(path, image)


def test_timing():
    samples = measure(lambda: 1, warmup=1, repeats=3)
    assert len(samples['samples_ms']) == 3
    assert samples['median_ms'] >= 0
    assert samples['iqr_ms'] >= 0
    for warmup, repeats in [(-1, 3), (0, 0), (True, 3), (0, 1.5)]:
        with pytest.raises(ValueError):
            measure(lambda: 1, warmup=warmup, repeats=repeats)


def test_json_infinity(tmp_path):
    path = tmp_path / 'metrics.json'
    write_json(path, {'psnr': float('inf'), 'identical_images': True})
    assert 'Infinity' not in path.read_text()
    assert json.loads(path.read_text()) == {
        'psnr': None, 'identical_images': True}
    with pytest.raises(FileExistsError):
        write_json(path, {})


def test_experiment_synthetic_software_check(tmp_path):
    pairs = tmp_path / 'pairs'
    pairs.mkdir()
    save_rgb(pairs / 'img_001_SRF_4_HR.png',
             np.full((24, 24, 3), 80, dtype=np.uint8))
    save_rgb(pairs / 'img_001_SRF_4_LR.png',
             np.full((6, 6, 3), 80, dtype=np.uint8))
    output = tmp_path / 'run'
    result = run_experiment(pairs, output, warmup=0, repeats=1)
    assert result['images'][0]['y']['mse'] == 0
    assert (output / 'metrics.json').is_file()
    assert (output / 'metrics.csv').is_file()
    assert (output / 'environment.json').is_file()
    assert (output / 'img_001_SRF_4_SR.png').is_file()
    with pytest.raises(FileExistsError):
        run_experiment(pairs, output, warmup=0, repeats=1)


def test_git_blob_hash_and_existing_bad_file(tmp_path):
    expected = 'ce013625030ba8dba906f756967f9e9ca394464a'
    assert git_blob_sha1(b'hello\n') == expected
    output = tmp_path / 'set5'
    output.mkdir()
    protected = output / 'img_001_SRF_4_HR.png'
    protected.write_bytes(b'existing file')
    with pytest.raises(ValueError, match='size'):
        download_set5(output, tmp_path / 'docs')
    assert protected.read_bytes() == b'existing file'


@pytest.mark.parametrize('settings', [{'repeats': 0}, {'warmup': -1},
                                    {'backend': 'unknown'}])
def test_invalid_experiment_settings_leave_no_output(tmp_path, settings):
    hr = tmp_path / 'hr.png'
    save_rgb(hr, np.full((24, 24, 3), 80, dtype=np.uint8))
    output = tmp_path / 'must_not_exist'
    with pytest.raises(ValueError):
        run_experiment(None, output, hr_path=hr, **settings)
    assert not output.exists()
