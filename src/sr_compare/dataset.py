"""Fixed Set5 download with local SHA-256 and independent Git blob checking."""
import hashlib
import json
import logging
from pathlib import Path
import urllib.request

from .io import load_rgb

COMMIT = '8f6dd8c1d20cb7e8792a7177b4f6fd677633f598'
PREFIX = ('https://raw.githubusercontent.com/jbhuang0604/SelfExSR/'
          f'{COMMIT}/data/Set5/image_SRF_4/')
TREE_URL = ('https://api.github.com/repos/jbhuang0604/SelfExSR/git/trees/'
            f'{COMMIT}?recursive=1')
SIZES = [(371264, 31162), (120453, 11287), (127529, 10390),
         (113787, 9058), (118848, 10816)]
EXPECTED = {f'img_{i:03d}_SRF_4_{kind}.png': sizes[j]
            for i, sizes in enumerate(SIZES, 1)
            for j, kind in enumerate(['HR', 'LR'])}
LOGGER = logging.getLogger(__name__)


def git_blob_sha1(content: bytes) -> str:
    """Hash a Git blob object, including its type/size header."""
    header = f'blob {len(content)}\0'.encode('ascii')
    return hashlib.sha1(header + content).hexdigest()


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': 'SISR-M1'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def download_set5(destination: Path, docs_dir: Path) -> list[dict]:
    """Download approved PNGs; verify existing files without replacement."""
    for name, size in EXPECTED.items():
        path = destination / name
        if path.exists() and path.stat().st_size != size:
            raise ValueError(f'Existing file size mismatch: {path}')
    tree = json.loads(_get(TREE_URL))
    if tree['sha'] != COMMIT or tree.get('truncated'):
        raise ValueError('GitHub tree identity/truncation check failed.')
    entries = {item['path']: item for item in tree['tree']}
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, size in EXPECTED.items():
        path = destination / name
        entry = entries[f'data/Set5/image_SRF_4/{name}']
        if entry['size'] != size:
            raise ValueError(f'Source size mismatch: {name}')
        existing = path.exists()
        content = path.read_bytes() if existing else _get(PREFIX + name)
        blob = git_blob_sha1(content)
        if len(content) != size or blob != entry['sha']:
            raise ValueError(f'Source size/blob hash mismatch: {name}')
        if not existing:
            with path.open('xb') as stream:
                stream.write(content)
        # Stop for grayscale/alpha/palette before applying any conversion.
        image = load_rgb(path)
        record = {'name': name, 'bytes': size,
                  'sha256_local': hashlib.sha256(content).hexdigest(),
                  'git_blob_sha1': blob, 'source_git_blob_sha1': entry['sha'],
                  'source_checksum_status': 'PASS', 'shape': list(image.shape),
                  'mode': 'RGB', 'url': PREFIX + name}
        records.append(record)
        LOGGER.info('%s %s bytes=%d RGB=%s source_blob=PASS sha256=%s',
                    'VERIFIED_EXISTING' if existing else 'DOWNLOADED', name,
                    size, image.shape, record['sha256_local'])
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest = ''.join(
        f"{r['sha256_local']}  {destination.as_posix()}/{r['name']}\n"
        for r in records)
    manifest_path = docs_dir / 'SET5_MANIFEST.sha256'
    if manifest_path.exists():
        if manifest_path.read_text(encoding='utf-8') != manifest:
            raise ValueError('Existing manifest differs; refusing to replace.')
    else:
        manifest_path.write_text(manifest, encoding='utf-8')
    provenance = (
        '# Set5 provenance\n\n'
        'License: **chưa xác định**; chỉ dùng học thuật nội bộ, '
        'không phân phối.\n\n'
        f'Source commit: `{COMMIT}`\n\nGit tree API: {TREE_URL}\n\n'
        'SHA-256 bên dưới tự tính: mốc toàn vẹn cục bộ, '
        'không phải hash nguồn công bố.\n'
        'Git blob SHA-1 được đối chiếu độc lập với GitHub tree '
        'ở commit cố định: PASS.\n'
        'Mode: mọi ảnh RGB, không chuyển mode/crop/resize khi tải.\n'
        'Công cụ tạo LR cụ thể: UNVERIFIED. '
        'README chỉ nói bicubic kernel downsampling.\n'
        'Nguồn tài liệu: https://github.com/jbhuang0604/SelfExSR/blob/'
        f'{COMMIT}/README.md\n\n'
        '```json\n' + json.dumps(records, indent=2, ensure_ascii=False)
        + '\n```\n')
    provenance_path = docs_dir / 'SET5_PROVENANCE.md'
    if provenance_path.exists():
        previous = provenance_path.read_text(encoding='utf-8')
        if not previous.startswith(provenance):
            raise ValueError(
                'Existing provenance differs; refusing replacement.')
    else:
        provenance_path.write_text(provenance, encoding='utf-8')
    return records
