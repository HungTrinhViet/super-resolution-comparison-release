"""Verify Set5 dimensions/content and report unresolved LR provenance."""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sr_compare.bicubic import quantize, resize
from sr_compare.dataset import COMMIT
from sr_compare.experiment import write_json
from sr_compare.io import load_pair
from sr_compare.verification import difference, screen_content


def main() -> None:
    """Run three independent checks; any FAIL stops later experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('data/set5/image_SRF_4'))
    parser.add_argument('--output', type=Path,
                        default=Path('docs/SET5_VERIFICATION.json'))
    parser.add_argument('--provenance', type=Path,
                        default=Path('docs/SET5_PROVENANCE.md'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    pairs = []
    dimensions = []
    for index in range(1, 6):
        name = f'img_{index:03d}_SRF_4'
        hr, lr = load_pair(args.data_dir / f'{name}_HR.png',
                           args.data_dir / f'{name}_LR.png')
        pairs.append((name, hr, lr))
        dimensions.append({
            'name': name, 'status': 'PASS',
            'hr_shape': list(hr.shape), 'lr_shape': list(lr.shape)})
    content = screen_content(pairs)
    degradation = []
    for name, hr, lr in pairs:
        for backend in ['numpy', 'pillow']:
            comparison = difference(lr, quantize(resize(hr, .25, backend)))
            degradation.append({'name': name, 'backend': backend,
                                'status': 'UNVERIFIED', **comparison})
    result = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'dimensions': {'status': 'PASS', 'pairs': dimensions},
        'content': content,
        'degradation_method': {
            'status': 'UNVERIFIED',
            'reason': ('README confirms bicubic downsampling but exact '
                       'tool/version and settings are unknown; '
                       'no independent MATLAB oracle.'),
            'source': ('https://github.com/jbhuang0604/SelfExSR/blob/'
                       f'{COMMIT}/README.md'),
            'comparisons': degradation},
    }
    write_json(args.output, result)
    negatives = len(content['negative_controls'])
    exceeded = content['negative_exceed_count']
    with args.provenance.open('a', encoding='utf-8') as stream:
        stream.write('\n## Pair verification\n\n'
                     f"UTC: {result['verified_utc']}\n\n"
                     'Dimensions: PASS (5 original pairs, no crop/resize).\n\n'
                     f"Content: {content['status']}; negative controls "
                     f'exceeding threshold: {exceeded}/{negatives}.\n\n'
                     'LR generator: UNVERIFIED. Pixel match alone cannot '
                     'establish provenance.\n\n'
                     f'Full evidence: `{args.output.as_posix()}` '
                     '(Git-visible document).\n')
    logging.info('DIMENSIONS PASS, CONTENT %s, METHOD UNVERIFIED',
                 content['status'])
    logging.info('Full verification values: %s', json.dumps(result, indent=2))
    if content['status'] != 'PASS':
        raise ValueError('Set5 content screening failed; inspect docs/SET5_VERIFICATION.json.')


if __name__ == '__main__':
    main()
