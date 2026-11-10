"""Lock or render separate exploratory low-gradient, mid-luminance crops."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sr_compare.comparison_crops_v2 import lock_v2, render_v2


def main() -> None:
    """Keep v2 outputs separate from the committed official v1 artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('lock', 'render'))
    parser.add_argument('--manifest', type=Path,
                        default=Path('docs/comparison/exploratory_v2/crop_manifest_v2.json'))
    parser.add_argument('--receipt', type=Path, default=Path('logs/m7_v2_crop_lock.log'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/comparison/m7_exploratory_v2'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = args.report or Path('logs')/f'm7_v2_{args.operation}_report.json'
    args.manifest.resolve().relative_to(ROOT/'docs/comparison/exploratory_v2')
    args.receipt.resolve().relative_to(ROOT/'logs')
    report.resolve().relative_to(ROOT/'logs')
    if report.exists():
        raise FileExistsError(report)
    if args.operation == 'lock':
        result = lock_v2(args.manifest, args.receipt)
    else:
        args.output_dir.resolve().relative_to(ROOT/'results/comparison/m7_exploratory_v2')
        result = render_v2(args.manifest, args.receipt, args.output_dir)
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
