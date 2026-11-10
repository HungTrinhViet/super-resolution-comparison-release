"""F-T1–F-T5 and alignment checks for the conditionally locked FSRCNN weights."""
from io import BytesIO
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
from scipy.io import loadmat, savemat
import torch
from torch.nn import functional as functional

from reference_fsrcnn_numpy import reference_forward, scatter_buffer
from sr_compare.fsrcnn import (
    CHECKPOINT, SOURCE_NOTICE, Fsrcnn, convert_weights,
    deconvolve, load_fsrcnn, shared_prelu, super_resolve,
)

SHAPES = [(1, 25, 56), (56, 1, 12), *[(12, 9, 12)]*4,
          (12, 1, 56), (1, 81, 56)]


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(4)
    print(SOURCE_NOTICE)
    return load_fsrcnn(CHECKPOINT)


def forward(model, image):
    with torch.inference_mode():
        return model(torch.from_numpy(image.astype(np.float32))[None, None])[0, 0].numpy()


def synthetic_arrays():
    arrays = {name: np.empty((8, 1), dtype=object)
              for name in ['weights_conv', 'biases_conv', 'prelu_conv']}
    for layer, shape in enumerate(SHAPES):
        arrays['weights_conv'][layer, 0] = np.zeros(shape, dtype=np.float64)
        arrays['biases_conv'][layer, 0] = np.zeros((shape[2] if layer < 7 else 1, 1),
                                                   dtype=np.float32)
        arrays['prelu_conv'][layer, 0] = (np.full((1, 1), .5, dtype=np.float32)
                                         if layer < 7 else np.empty((1, 0)))
    return arrays


def test_f_t1_cells_and_parameters(model):
    arrays = loadmat(CHECKPOINT)
    assert len(model.stages) == 7
    assert sum(p.numel() for p in model.parameters()) == 12644
    assert len(model.state_dict()) == 23
    for layer in range(7):
        np.testing.assert_array_equal(model.stages[layer].conv.bias.detach().numpy(),
                                      arrays['biases_conv'][layer, 0].ravel())
        np.testing.assert_array_equal(model.stages[layer].slope.detach().numpy(),
                                      arrays['prelu_conv'][layer, 0].ravel())
    assert arrays['prelu_conv'][7, 0].shape == (1, 0)
    assert sum(stage.slope.numel() for stage in model.stages) == 7
    assert model.deconv_weight.shape == (56, 1, 9, 9)


@pytest.mark.parametrize('fault', ['cell_shape', 'weight_dtype', 'bias_dtype',
                                 'empty_prelu', 'nonfinite'])
def test_f_t1_rejects_changed_schema(fault):
    arrays = synthetic_arrays()
    if fault == 'cell_shape':
        arrays['weights_conv'] = arrays['weights_conv'].T
    elif fault == 'weight_dtype':
        arrays['weights_conv'][0, 0] = arrays['weights_conv'][0, 0].astype(np.float32)
    elif fault == 'bias_dtype':
        arrays['biases_conv'][0, 0] = arrays['biases_conv'][0, 0].astype(np.float64)
    elif fault == 'empty_prelu':
        arrays['prelu_conv'][7, 0] = np.zeros((1, 1), dtype=np.float32)
    else:
        arrays['weights_conv'][0, 0][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        convert_weights(arrays)


def test_f_t1_rejects_hash_before_loading(tmp_path, monkeypatch):
    path = tmp_path/'changed.mat'
    path.write_bytes(b'not the locked checkpoint')

    def forbidden_load(*args, **kwargs):
        pytest.fail('loadmat must not run before the SHA-256 check.')

    monkeypatch.setattr('sr_compare.fsrcnn.loadmat', forbidden_load)
    with pytest.raises(ValueError, match='SHA-256'):
        load_fsrcnn(path)


def test_f_t2_all_output_pixels(model):
    samples = {'noise': np.random.default_rng(72).random((11, 15)),
               'gradient': np.linspace(0, 1, 13*17).reshape(13, 17),
               'checker': (np.indices((12, 19)).sum(axis=0) % 2).astype(float),
               'constant': np.full((9, 14), .4)}
    with zipfile.ZipFile('data/quarantine/srcnn/SRCNN_v1.zip') as archive:
        for name in ['zebra', 'monarch']:
            with Image.open(BytesIO(archive.read(f'SRCNN/Set14/{name}.bmp'))) as image:
                samples[name] = np.array(image)[30:42, 41:58].mean(axis=-1)/255
    for name, image in samples.items():
        actual, expected = forward(model, image), reference_forward(CHECKPOINT, image)
        error = np.abs(actual-expected)
        a = np.clip(np.floor(actual.astype(float)*255+.5), 0, 255).astype(np.uint8)
        b = np.clip(np.floor(expected*255+.5), 0, 255).astype(np.uint8)
        uint_error = np.abs(a.astype(int)-b.astype(int))
        print('F-T2', name, 'max/mean float', error.max(), error.mean(),
              'uint8 pixels different', int((uint_error != 0).sum()),
              '/', uint_error.size, 'uint8 max', uint_error.max())
        assert actual.shape == expected.shape
        assert error.max() <= 1e-4


def test_f_t3_layout_negative_controls(tmp_path):
    arrays = synthetic_arrays()
    arrays['weights_conv'][0, 0][0, 1+5*3, 0] = 1
    arrays['weights_conv'][0, 0][0, 3+5*0, 1] = 1
    for layer in range(1, 7):
        size = int(np.sqrt(SHAPES[layer][1]))
        for channel in (0, 1):
            arrays['weights_conv'][layer, 0][channel, size//2+size*(size//2), channel] = 1
    arrays['weights_conv'][7, 0][0, 1+9*7, 0] = 1
    arrays['weights_conv'][7, 0][0, 6+9*2, 1] = 1
    path = tmp_path/'synthetic_layout_only.mat'
    savemat(path, arrays)
    image = np.zeros((9, 11))
    image[4, 5] = 1
    image[5, 3] = 1
    expected = reference_forward(path, image)
    state = convert_weights(arrays)
    network = Fsrcnn()
    network.load_state_dict(state, strict=True)
    np.testing.assert_array_equal(forward(network, image), expected)
    for name in ['row_major', 'flipped_kernel', 'wrong_channel_layout']:
        wrong = {key: value.clone() for key, value in state.items()}
        if name == 'row_major':
            for channel in (0, 1):
                wrong['stages.0.conv.weight'][channel, 0] = torch.tensor(
                    arrays['weights_conv'][0, 0][0, :, channel].reshape(5, 5),
                    dtype=torch.float32)
                wrong['deconv_weight'][channel, 0] = torch.tensor(
                    arrays['weights_conv'][7, 0][0, :, channel].reshape(9, 9),
                    dtype=torch.float32)
        elif name == 'flipped_kernel':
            wrong['stages.0.conv.weight'] = wrong['stages.0.conv.weight'].flip(-1, -2)
            wrong['deconv_weight'] = wrong['deconv_weight'].flip(-1, -2)
        else:
            wrong['deconv_weight'][[0, 1]] = wrong['deconv_weight'][[1, 0]]
        network.load_state_dict(wrong, strict=True)
        delta = np.max(np.abs(forward(network, image)-expected))
        print('F-T3 negative control', name, 'max deviation', delta)
        assert delta > 1e-4


def test_f_t4_hand_calculated_conv_prelu_and_final_bias(model):
    isolated = Fsrcnn()
    for layer, stage in enumerate(isolated.stages, 1):
        second_channel = min(1, stage.conv.in_channels-1)
        second_value = .5 if second_channel else -.25
        with torch.no_grad():
            stage.conv.weight.zero_()
            stage.conv.weight[0, 0].fill_(.125)
            stage.conv.weight[1, second_channel].fill_(.25)
            stage.conv.bias.zero_()
            stage.conv.bias[:2].copy_(torch.tensor([.125, -.25]))
            stage.slope.fill_(-.5)
        image = torch.zeros(1, stage.conv.in_channels, 5, 7)
        image[:, 0] = -.25
        image[:, second_channel] = second_value
        area = stage.conv.kernel_size[0]**2
        z = np.zeros(stage.conv.out_channels)
        z[0] = -.25*.125*area + .125
        z[1] = second_value*.25*area - .25
        manual = np.maximum(z, 0) - .5*np.minimum(z, 0)
        actual = stage(image).detach().numpy()
        expected = np.broadcast_to(manual[None, :, None, None], actual.shape)
        error = np.max(np.abs(actual-expected))
        print('F-T4 hand layer', layer, 'kernel', stage.conv.kernel_size[0],
              'padding', stage.padding, 'max error', error)
        assert error <= 1e-6
    for layer in (0, 6):
        slope = model.stages[layer].slope.detach()
        negative = torch.tensor([[[[-2.]], [[-4.]]]])
        np.testing.assert_allclose(shared_prelu(negative, slope).detach().numpy(),
                                   (negative*slope).numpy(), atol=1e-6, rtol=0)
        assert shared_prelu(negative, slope).min() > 0
        print('F-T4 author shared negative slope layer', layer+1, slope.item())
    with torch.no_grad():
        isolated.deconv_weight.zero_()
        isolated.deconv_bias.fill_(-.25)
    np.testing.assert_array_equal(forward(isolated, np.zeros((3, 5))), -.25)
    print('F-T4 seven shared slopes; final negative bias retained')


def test_f_t5_interior_and_manual_replicate_boundary():
    features = torch.zeros(1, 1, 5, 7)
    features[0, 0, 2, 3] = 1
    kernel = torch.zeros(1, 1, 9, 9)
    kernel[0, 0, 1, 7] = 1
    result = deconvolve(features, kernel, torch.zeros(1)).numpy()[0, 0]
    assert np.array_equal(np.argwhere(result != 0), [[4*2+1-4, 4*3+7-4]])
    for label, position, target, expected in [
            ('corner', (0, 0), (0, 0), 4), ('top_edge', (0, 1), (0, 4), 2)]:
        features = torch.zeros(1, 1, 3, 4)
        features[0, 0, position[0], position[1]] = 1
        kernel = torch.ones(1, 1, 9, 9)
        actual = deconvolve(features, kernel, torch.zeros(1))[0, 0]
        unpadded = functional.conv_transpose2d(features, kernel, stride=4,
                                              padding=4, output_padding=3)[0, 0]
        assert actual[target] == expected and unpadded[target] == 1
        print('F-T5 manual', label, 'replicate sum', actual[target].item(),
              'without replicate', unpadded[target].item())


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (5, 1), (5, 7)])
def test_f_t5_full_buffer_crop_and_negative_controls(shape):
    features = torch.ones(1, 1, *shape)
    kernels = torch.ones(1, 1, 9, 9)
    padded = functional.pad(features, (1, 1, 1, 1), mode='replicate')
    natural = functional.conv_transpose2d(padded, kernels, stride=4)
    reference = scatter_buffer(features.numpy()[0].astype(float), kernels.numpy()[:, 0])
    h, w = shape
    np.testing.assert_array_equal(natural.numpy()[0, 0], reference[:4*h+13, :4*w+13])
    expected = reference[8:8+4*h, 8:8+4*w]
    actual = deconvolve(features, kernels, torch.zeros(1)).numpy()[0, 0]
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (4*h, 4*w)
    assert actual.min() != actual.max()  # Stride phases need not preserve constants.
    unpadded = functional.conv_transpose2d(features, kernels, stride=4)
    controls = {
        'crop7': natural[0, 0, 7:7+4*h, 7:7+4*w],
        'crop9': natural[0, 0, 9:9+4*h, 9:9+4*w],
        'no_replicate': unpadded[0, 0, 4:4+4*h, 4:4+4*w],
        'padding4_output_padding3': functional.conv_transpose2d(
            features, kernels, stride=4, padding=4, output_padding=3)[0, 0],
    }
    for label, value in controls.items():
        error = np.max(np.abs(value.numpy()-expected))
        print('F-T5 negative', shape, label, 'max deviation', error)
        assert error > 0


def test_f_adapter_and_nonfinite(model):
    image = np.full((5, 7, 3), [10, 120, 230], dtype=np.uint8)
    result = super_resolve(model, image)
    assert result.rgb.shape == (20, 28, 3) and result.rgb.dtype == np.uint8
    assert result.y_uint8.shape == result.y_float.shape == (20, 28)
    assert result.source_notice == SOURCE_NOTICE
    class Nonfinite(torch.nn.Module):
        def forward(self, image):
            return torch.full((1, 1, image.shape[2]*4, image.shape[3]*4), float('nan'))
    with pytest.raises(ValueError, match='finite'):
        super_resolve(Nonfinite(), image)


def test_f_t8_alignment_tool():
    spec = importlib.util.spec_from_file_location('sanity_fsrcnn', 'scripts/sanity_fsrcnn_set14.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = np.zeros((64, 80))
    image[16:48, 18:62] = np.random.default_rng(18).random((32, 44))
    moved = np.zeros_like(image)
    moved[3:, :-2] = image[:-3, 2:]
    assert module.displacement(image, moved) == (3, -2)
    assert module.displacement(image, image) == (0, 0)
