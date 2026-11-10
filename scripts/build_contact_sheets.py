"""Lock HR-only crop rules before rendering any real Set5 contact sheets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sr_compare.comparison_crops import lock_manifest, render_sheets


def main() -> None:
    """Keep selection isolated from SR/metric reads; render only from the lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('lock', 'render'))
    parser.add_argument('--manifest', type=Path, default=Path('docs/comparison/crop_manifest.json'))
    parser.add_argument('--receipt', type=Path, default=Path('logs/m7_crop_lock.log'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/comparison/m7_v1'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = args.report or Path('logs')/f'm7_sheet_{args.operation}_report.json'
    args.manifest.resolve().relative_to(ROOT/'docs/comparison')
    args.receipt.resolve().relative_to(ROOT/'logs')
    report.resolve().relative_to(ROOT/'logs')
    if report.exists():
        raise FileExistsError(f'Report already exists: {report}')
    if args.operation == 'lock':
        result = lock_manifest(args.manifest, args.receipt, ROOT/'data/set5/image_SRF_4')
    else:
        args.output_dir.resolve().relative_to(ROOT/'results/comparison/m7_v1')
        result = render_sheets(args.manifest, args.receipt, args.output_dir)
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
