"""HR-only integer crop selection, immutable receipts and read-only contact sheets."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_VERSION = 'hr_integer_three_rules_v1'
RULES = ('hr_gradient64_v1', 'hr_lowgrad64_v1', 'center64_v1')
CODE = ('src/sr_compare/comparison_crops.py', 'scripts/build_contact_sheets.py')


def digest(path: Path) -> str:
    """Hash exact file bytes without decoding."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer_y(rgb: np.ndarray) -> np.ndarray:
    """Quantized BT.601 for crop features only, never for reference metrics."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Crop selection requires RGB uint8.')
    values = rgb.astype(np.int64)
    weighted = 65481*values[..., 0]+128553*values[..., 1]+24966*values[..., 2]
    return 16+(weighted+127500)//255000


def window_sums(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Sum all valid windows using exact int64 integral sums."""
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (integral[height:, width:]-integral[:-height, width:]
            -integral[height:, :-width]+integral[:-height, :-width])


def select_crops(rgb: np.ndarray) -> list[dict]:
    """Select high-gradient, eligible low-gradient and center windows from HR only."""
    y = integer_y(rgb)
    if min(y.shape) < 64:
        raise ValueError('HR must be at least64x64; no crop-size fallback.')
    scores = (window_sums(np.diff(y, axis=1)**2, 64, 63)
              + window_sums(np.diff(y, axis=0)**2, 63, 64))
    sums, squares = window_sums(y, 64, 64), window_sums(y*y, 64, 64)
    variance = 4096*squares-sums*sums
    eligible = variance >= 4*4096**2
    if not eligible.any():
        raise ValueError('No eligible low-gradient window with population std>=2.')
    high = np.unravel_index(scores.argmax(), scores.shape)
    low = np.unravel_index(np.where(eligible, scores, np.iinfo(np.int64).max).argmin(), scores.shape)
    center = ((y.shape[0]-64)//2, (y.shape[1]-64)//2)
    coordinates = (high, low, center)
    counts = Counter(coordinates)
    records = []
    for rule, (row, column) in zip(RULES, coordinates):
        records.append({
            'rule': rule, 'rule_version': 1, 'algorithm_version': ALGORITHM_VERSION,
            'x': int(column), 'y': int(row), 'w': 64, 'h': 64, 'enlargement': 4,
            'gradient_score': int(scores[row, column]),
            'sum_y': int(sums[row, column]), 'sum_y2': int(squares[row, column]),
            'variance_numerator': int(variance[row, column]), 'population_n': 4096,
            'std_condition_required': rule == 'hr_lowgrad64_v1',
            'std_condition_met': bool(eligible[row, column]),
            'std_threshold_numerator': 4*4096**2,
            'tie_break': 'smallest_y_then_x',
            'duplicate_coordinates': counts[(row, column)] > 1,
        })
    return records


def read_rgb(path: Path) -> np.ndarray:
    """Read RGB pixels without mode, ICC or brightness conversion."""
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'Expected RGB: {path}')
        return np.array(image)


def save_json(path: Path, value: dict) -> None:
    """Write canonical UTF-8 metadata exclusively."""
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def lock_manifest(manifest: Path, receipt: Path, hr_dir: Path) -> dict:
    """Read only five HR PNGs and selector code before creating the crop lock."""
    if manifest.exists() or receipt.exists():
        raise FileExistsError('Crop manifest or receipt already exists.')
    records, hashes = [], {}
    for index in range(1, 6):
        name = f'img_{index:03d}_SRF_4'
        path = hr_dir/f'{name}_HR.png'
        relative = path.resolve().relative_to(ROOT).as_posix()
        rgb = read_rgb(path)
        hashes[relative] = digest(path)
        for crop in select_crops(rgb):
            records.append({**crop, 'image_id': name, 'hr_name': path.name,
                            'hr_path': relative, 'hr_sha256': hashes[relative],
                            'hr_width': rgb.shape[1], 'hr_height': rgb.shape[0]})
    locked = datetime.now(timezone.utc).isoformat()
    records = [{**record, 'locked_utc': locked} for record in records]
    data = {'schema_version': 1, 'algorithm_version': ALGORITHM_VERSION,
            'locked_utc': locked, 'crop_feature_only_not_metric': True,
            'historical_sr_viewing_disclosed': True, 'records': records}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest, data)
    capture = {'locked_utc': locked, 'manifest_sha256': digest(manifest),
               'source_sha256': {name: digest(ROOT/name) for name in CODE},
               'hr_sha256': hashes}
    save_json(receipt, capture)
    return capture


def verify_manifest(manifest: Path, receipt: Path) -> dict:
    """Require unchanged bytes and valid coordinates before reading any SR."""
    captured = json.loads(receipt.read_text(encoding='utf-8'))
    if digest(manifest) != captured['manifest_sha256']:
        raise ValueError('Locked crop manifest hash changed.')
    if set(captured['source_sha256']) != set(CODE):
        raise ValueError('Incomplete selector source receipt.')
    for name, expected in {**captured['source_sha256'], **captured['hr_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Locked source/HR changed: {name}')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if len(data['records']) != 15 or data['locked_utc'] != captured['locked_utc']:
        raise ValueError('Invalid locked crop count or UTC.')
    for index in range(1, 6):
        rows = [r for r in data['records'] if r['image_id'] == f'img_{index:03d}_SRF_4']
        if len(rows) != 3 or {r['rule'] for r in rows} != set(RULES):
            raise ValueError('Expected exactly three rules per image.')
        for row in rows:
            if (row['w'] != 64 or row['h'] != 64 or row['enlargement'] != 4
                    or row['x'] < 0 or row['y'] < 0
                    or row['x']+64 > row['hr_width'] or row['y']+64 > row['hr_height']):
                raise ValueError('Invalid crop bounds.')
    return data


def render_panel(images: list[tuple[str, str, np.ndarray]], name: str,
                 crop: dict | None) -> Image.Image:
    """Paste native full images or exact nearest x4 crops below external labels."""
    width = 256 if crop else images[0][2].shape[1]
    height = 256 if crop else images[0][2].shape[0]
    canvas = Image.new('RGB', (6*(width+8)+8, height+140), 'white')
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=14)
    for column, (label, run_id, rgb) in enumerate(images):
        image = Image.fromarray(rgb)
        if crop:
            x, y = crop['x'], crop['y']
            image = image.crop((x, y, x+64, y+64)).resize((256, 256), Image.Resampling.NEAREST)
        x = 8+column*(width+8)
        lines = [name[4:7]+' '+(crop['rule'] if crop else 'full native'), label,
                 *textwrap.wrap(run_id, max(20, width//8))]
        for row, line in enumerate(lines):
            draw.text((x, 5+row*19), line, font=font, fill='black')
        canvas.paste(image, (x, 132))
    return canvas


def render_sheets(manifest: Path, receipt: Path, output: Path) -> dict:
    """Render from a verified lock; preserve source bytes and record pixel hashes."""
    from .comparison import SOURCES, load_runs, verify_source_snapshot

    data = verify_manifest(manifest, receipt)
    load_runs(ROOT)
    verify_source_snapshot(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    (output/'full').mkdir()
    (output/'crops').mkdir()
    sources, hashes = {}, {}
    started = datetime.now(timezone.utc).isoformat()
    if datetime.fromisoformat(started) <= datetime.fromisoformat(data['locked_utc']):
        raise ValueError('Sheet rendering must start after crop lock.')
    for index in range(1, 6):
        name = f'img_{index:03d}_SRF_4'
        crops = [r for r in data['records'] if r['image_id'] == name]
        hr_path = ROOT/crops[0]['hr_path']
        hr = read_rgb(hr_path)
        sources[hr_path.as_posix()] = digest(hr_path)
        images = [('HR', 'SelfExSR HR '+digest(hr_path)[:12], hr)]
        for algorithm, run_id, _ in SOURCES:
            path = ROOT/'results'/run_id/f'{name}_SR.png'
            rgb = read_rgb(path)
            if rgb.shape != hr.shape:
                raise ValueError(f'SR/HR shape mismatch: {algorithm}/{name}')
            sources[path.as_posix()] = digest(path)
            images.append((algorithm, run_id, rgb))
        for crop in (None, *crops):
            relative = (f'full/full_{name}.png' if crop is None
                        else f"crops/crop_{name}_{crop['rule']}.png")
            canvas = render_panel(images, name, crop)
            canvas.save(output/relative, compress_level=9, optimize=False)
            hashes[relative] = {'sha256': digest(output/relative),
                                'pixel_sha256': hashlib.sha256(canvas.tobytes()).hexdigest(),
                                'shape': [canvas.height, canvas.width, 3], 'dtype': 'uint8'}
    for name, expected in sources.items():
        if digest(Path(name)) != expected:
            raise ValueError('Source PNG changed during rendering.')
    verify_source_snapshot(ROOT)
    return {'render_started_utc': started, 'source_count': len(sources),
            'manifest_sha256': digest(manifest), 'images': hashes}
