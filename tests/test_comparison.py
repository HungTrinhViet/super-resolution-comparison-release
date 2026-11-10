"""Synthetic checks for read-only comparison and deterministic SVG output."""
import hashlib
import json

import numpy as np
import pytest

from sr_compare.comparison import load_runs, summarize, write_outputs


def fixture_runs(tmp_path):
    sources = []
    for index, label in enumerate(('Bicubic', 'SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN')):
        run = f'synthetic_{label}'
        path = tmp_path/'results'/run
        path.mkdir(parents=True)
        rows = []
        for image in range(1, 6):
            timing = {'warmup': 3, 'repeats': 30 if index == 0 else 10}
            samples = list(range(1, timing['repeats']+1))
            q1, q3 = np.percentile(samples, [25, 75])
            timing.update(samples_ms=samples, median_ms=float(np.median(samples)),
                          iqr_ms=float(q3-q1))
            row = {'name': f'img_{image:03d}_SRF_4',
                   'hr_shape': [256, 256, 3], 'lr_shape': [64, 64, 3],
                   'y': {'psnr': 24+index+image*.000004, 'ssim': .7+index*.01},
                   'rgb': {'psnr': 22+index+image*.000004, 'ssim': .6+index*.01}}
            if index == 0:
                row['timing'] = timing
            else:
                row.update(timing_pipeline=timing, timing_forward=timing)
            rows.append(row)
        means = {s: {m: float(np.mean([r[s][m] for r in rows]))
                     for m in ('psnr', 'ssim')} for s in ('y', 'rgb')}
        data = {'scale': 4, 'images': rows, 'mean_per_image': means}
        if index == 0:
            data['backend'] = 'numpy'
        else:
            data['run_id'] = run
        raw = json.dumps(data).encode()
        (path/'metrics.json').write_bytes(raw)
        sources.append((label, run, hashlib.sha256(raw).hexdigest()))
    return sources


def test_complete_means_and_deterministic_output(tmp_path):
    sources = fixture_runs(tmp_path)
    records = load_runs(tmp_path, sources)
    assert len(records) == 25
    means = summarize(records)
    assert means[0]['y_psnr'] == pytest.approx(24.000012)
    assert means[0]['y_psnr'] != 24.
    write_outputs(records, tmp_path/'first')
    write_outputs(records, tmp_path/'second')
    for file in (tmp_path/'first').rglob('*'):
        if file.is_file():
            assert file.read_bytes() == (tmp_path/'second'/file.relative_to(tmp_path/'first')).read_bytes()
    assert len(list((tmp_path/'first'/'figures').glob('*.svg'))) == 6
    assert 'Bicubic' not in (tmp_path/'first'/'tables'/'timing_network.csv').read_text()
    parent = tmp_path/'lock_first'
    parent.mkdir()
    (parent/'crop_manifest.json').write_text('synthetic locked metadata')
    write_outputs(records, parent)
    assert (parent/'crop_manifest.json').read_text() == 'synthetic locked metadata'
    with pytest.raises(FileExistsError):
        write_outputs(records, tmp_path/'first')


@pytest.mark.parametrize('fault', ['hash', 'missing', 'run_id', 'mean', 'timing'])
def test_reject_bad_sources(tmp_path, fault):
    sources = fixture_runs(tmp_path)
    label, run, digest = sources[1]
    path = tmp_path/'results'/run/'metrics.json'
    data = json.loads(path.read_text())
    if fault == 'hash':
        path.write_bytes(b'changed')
    else:
        if fault == 'missing':
            del data['images'][0]['y']['ssim']
        elif fault == 'run_id':
            data['run_id'] = 'wrong'
        elif fault == 'mean':
            data['mean_per_image']['y']['psnr'] += .01
        else:
            data['images'][0]['timing_forward']['samples_ms'].pop()
        raw = json.dumps(data).encode()
        path.write_bytes(raw)
        sources[1] = (label, run, hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        load_runs(tmp_path, sources)


def test_asymmetric_svg_intervals_are_quartiles_not_median_plus_iqr():
    import math
    import re
    from xml.etree import ElementTree
    from sr_compare.comparison import timing_record
    from sr_compare.comparison_plots import chart
    samples = [1, 1, 2, 3, 4, 8, 20, 30, 50, 100]
    q1, q3 = np.percentile(samples, [25, 75])
    median = float(np.median(samples))
    assert median != (q1+q3)/2
    timing = timing_record({'samples_ms': samples, 'warmup': 3, 'repeats': 10,
                            'median_ms': median, 'iqr_ms': float(q3-q1)}, 10)
    rows = [{'algorithm': algorithm, 'image_id': f'img_{i:03d}_SRF_4',
             'pipeline': timing, 'network': None if algorithm == 'Bicubic' else timing}
            for algorithm in ('Bicubic', 'SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN')
            for i in range(1, 6)]
    root = ElementTree.fromstring(chart(rows, 'unit only', scope='network'))
    ns = {'s': 'http://www.w3.org/2000/svg'}
    circle = root.find('.//s:circle', ns)
    assert float(circle.attrib['cy']) == pytest.approx(545-math.log10(median)/2*410, abs=.001)
    path = next(p for p in root.findall('.//s:path', ns) if p.attrib.get('stroke') == '#E69F00')
    match = re.match(r'M[\d.]+ ([\d.]+) V([\d.]+)', path.attrib['d'])
    assert float(match[1]) == pytest.approx(545-math.log10(q3)/2*410, abs=.001)
    assert float(match[2]) == pytest.approx(545-math.log10(q1)/2*410, abs=.001)
    assert not root.findall('.//s:image', ns)


def test_source_snapshot_rejects_replaced_png(tmp_path):
    from sr_compare.comparison import verify_source_snapshot
    source = tmp_path/'unit.png'
    source.write_bytes(b'synthetic source bytes')
    path = tmp_path/'docs/comparison/manifests/sources.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'unit.png': hashlib.sha256(source.read_bytes()).hexdigest()}))
    receipt = tmp_path/'logs/m7_sources_lock.log'
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({'source_snapshot_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    verify_source_snapshot(tmp_path)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source artifact'):
        verify_source_snapshot(tmp_path)
