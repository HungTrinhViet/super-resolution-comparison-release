"""T1–T5: author weights, independent forward, layout and adapter checks."""
from io import BytesIO
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
from scipy.io import loadmat
import torch

from reference_srcnn_numpy import reference_forward
from sr_compare.srcnn import Srcnn, convert_weights, load_srcnn, super_resolve

CHECKPOINT = Path('data/checkpoints/srcnn/SRCNN/model/9-5-5(ImageNet)/x4.mat')


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print('SRCNN test CPU threads:', torch.get_num_threads(),
          torch.get_num_interop_threads())


def real_crop(name):
    with zipfile.ZipFile('data/quarantine/srcnn/SRCNN_v1.zip') as archive:
        with Image.open(BytesIO(archive.read(f'SRCNN/Set14/{name}.bmp'))) as image:
            return np.asarray(image)[30:42, 41:58].copy()


def forward(model, image):
    with torch.inference_mode():
        return model(torch.from_numpy(image.astype(np.float32))[None, None])[0, 0].numpy()


def test_t1_author_weights():
    arrays = loadmat(CHECKPOINT)
    model = load_srcnn(CHECKPOINT)
    assert sum(p.numel() for p in model.parameters()) == 57281
    for i in range(1, 4):
        np.testing.assert_array_equal(getattr(model, f'conv{i}').bias.detach().numpy(),
                                      arrays[f'biases_conv{i}'].ravel().astype(np.float32))


def test_t2_independent_forward_samples():
    samples = {'noise': np.random.default_rng(7).random((11, 15)),
               'gradient': np.linspace(0, 1, 13*17).reshape(13, 17),
               'checker': (np.indices((12, 19)).sum(axis=0) % 2).astype(float),
               'constant': np.full((9, 14), .4),
               'edge': np.pad(np.ones((10, 2)), ((0, 0), (0, 13)))}
    for name in ['zebra', 'monarch']:
        samples[name] = real_crop(name).mean(axis=2)/255
    model = load_srcnn(CHECKPOINT)
    for name, sample in samples.items():
        error = np.abs(forward(model, sample) - reference_forward(CHECKPOINT, sample))
        print('T2', name, 'max', error.max(), 'mean', error.mean())
        assert error.max() <= 1e-4


def synthetic_arrays():
    return {'weights_conv1': np.zeros((81, 64)), 'biases_conv1': np.zeros((64, 1)),
            'weights_conv2': np.zeros((64, 25, 32)), 'biases_conv2': np.zeros((32, 1)),
            'weights_conv3': np.zeros((32, 25)), 'biases_conv3': np.zeros((1, 1))}


def test_t3_column_major_impulse():
    arrays = synthetic_arrays()
    arrays['weights_conv1'][2 + 9*6, 0] = 1
    arrays['weights_conv2'][0, 2 + 5*2, 0] = 1
    arrays['weights_conv3'][0, 2 + 5*2] = 1
    model = Srcnn()
    model.load_state_dict(convert_weights(arrays), strict=True)
    image = np.zeros((25, 29))
    image[12, 14] = 1
    result = forward(model, image)
    assert np.unravel_index(result.argmax(), result.shape) == (14, 12)
    assert result[10, 16] == 0  # Incorrect row-major mapping predicts this position.
    wrong = convert_weights(arrays)
    wrong['conv1.weight'].zero_()
    wrong['conv1.weight'][0, 0] = torch.tensor(
        arrays['weights_conv1'][:, 0].reshape(9, 9), dtype=torch.float32)
    model.load_state_dict(wrong, strict=True)
    row_major = forward(model, image)
    assert np.unravel_index(row_major.argmax(), row_major.shape) == (10, 16)
    assert not np.array_equal(row_major, result)


def test_t4_constant_hand_calculation():
    arrays = synthetic_arrays()
    arrays['weights_conv1'][:, 0] = .01
    arrays['weights_conv2'][0, :, 0] = .02
    arrays['weights_conv3'][0, :] = .03
    arrays['biases_conv1'][0, 0] = .1
    arrays['biases_conv2'][0, 0] = .2
    arrays['biases_conv3'][0, 0] = .3
    model = Srcnn()
    model.load_state_dict(convert_weights(arrays), strict=True)
    expected = max(max(.4*81*.01 + .1, 0)*25*.02 + .2, 0)*25*.03 + .3
    error = np.max(np.abs(forward(model, np.full((11, 15), .4)) - expected))
    print('T4 max error', error)
    assert error <= 1e-6


def test_t5_adapter_shapes_and_finite_guard():
    model = load_srcnn(CHECKPOINT)
    for lr in [np.full((5, 7, 3), 128, dtype=np.uint8), real_crop('monarch')]:
        result = super_resolve(model, lr)
        assert result.rgb.shape == (lr.shape[0]*4, lr.shape[1]*4, 3)
        assert result.rgb.dtype == result.y_uint8.dtype == np.uint8
        assert np.isfinite(result.y_float).all()
    with torch.no_grad():
        model.conv3.bias.fill_(float('nan'))
    with pytest.raises(ValueError, match='finite'):
        super_resolve(model, np.zeros((5, 7, 3), dtype=np.uint8))


def test_t9_alignment_tool():
    path = Path('scripts/sanity_srcnn_set14.py')
    spec = importlib.util.spec_from_file_location('sanity_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = np.zeros((64, 80))
    image[16:48, 18:62] = np.random.default_rng(18).random((32, 44))
    shifted = np.zeros_like(image)
    shifted[3:, :-2] = image[:-3, 2:]
    assert module.displacement(image, shifted) == (3, -2)
    assert module.displacement(image, image) == (0, 0)
