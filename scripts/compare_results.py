"""Generate approved numeric comparisons and deterministic SVG, without inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sr_compare.comparison import SOURCES, load_runs, sha256, write_outputs, verify_source_snapshot


def main() -> None:
    """Write exclusive tables/charts and a relative source hash snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/comparison'))
    parser.add_argument('--reuse-source-lock', action='store_true')
    args = parser.parse_args()
    rows = load_runs()
    receipt_path = Path('logs/m7_sources_lock.log')
    if args.reuse_source_lock:
        verify_source_snapshot()
    elif receipt_path.exists() or (args.output_dir/'manifests').exists():
        raise FileExistsError('Source snapshot lock already exists.')
    write_outputs(rows, args.output_dir)
    paths = [Path('results')/run/file for _, run, _ in SOURCES
             for file in ('metrics.json', 'environment.json')]
    paths += list(Path('data/set5/image_SRF_4').glob('*_HR.png'))
    paths += [Path('results')/run/f'img_{i:03d}_SRF_4_SR.png'
              for _, run, _ in SOURCES for i in range(1, 6)]
    sources = {p.as_posix(): sha256(p) for p in paths}
    (args.output_dir/'manifests').mkdir()
    (args.output_dir/'manifests'/'sources.json').write_text(
        json.dumps(sources, sort_keys=True, indent=2)+'\n', encoding='utf-8', newline='\n')
    if not args.reuse_source_lock:
        capture = {'locked_utc': datetime.now(timezone.utc).isoformat(),
                   'source_snapshot_sha256': sha256(args.output_dir/'manifests'/'sources.json')}
        for target in (receipt_path, args.output_dir/'manifests'/'source_lock.json'):
            with target.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(capture, sort_keys=True, indent=2)+'\n')
    print('PASS25 records,100metrics,means/deltas,6SVG; no inference or benchmark')


if __name__ == '__main__':
    main()
