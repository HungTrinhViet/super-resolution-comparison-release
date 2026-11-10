"""Deterministic numeric-only SVG charts with explicit units and timing scopes."""
from html import escape
import json
import math
from pathlib import Path

COLORS = ('#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7')
ALGORITHMS = ('Bicubic', 'SRCNN', 'FSRCNN', 'EDSR', 'ESRGAN')


def text(x: float, y: float, value: str, size: int = 15, anchor: str = 'start') -> str:
    """Escape every label rather than embedding arbitrary SVG fragments."""
    return (f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" '
            f'text-anchor="{anchor}">{escape(value)}</text>')


def chart(rows: list[dict], title: str, metric: str | None = None,
          scope: str | None = None, means: bool = False) -> str:
    """Render grouped metric bars or asymmetric quartile intervals on log-ms axes."""
    timing = scope is not None
    algorithms = ALGORITHMS if scope != 'network' else ALGORITHMS[1:]
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="740" viewBox="0 0 1100 740">',
              '<rect width="1100" height="740" fill="white"/>',
              '<g font-family="sans-serif" fill="black">', text(55, 35, title, 23)]
    metadata = [{k: v for k, v in row.items() if k != ('network' if scope == 'pipeline' else 'pipeline')}
                for row in rows]
    pieces.append('<metadata>'+escape(json.dumps(metadata, sort_keys=True, separators=(',', ':')))+' </metadata>')
    for index, algorithm in enumerate(algorithms):
        color = COLORS[ALGORITHMS.index(algorithm)]
        x = 75+index*195
        pieces.extend([f'<rect x="{x}" y="59" width="17" height="17" fill="{color}"/>',
                       text(x+24, 73, algorithm)])
    if timing:
        values = [r[scope][key] for r in rows if r[scope] is not None
                  for key in ('q25_ms', 'q75_ms')]
        lower = math.floor(math.log10(min(values)))
        upper = max(lower+1, math.ceil(math.log10(max(values))))
        ticks = [(10**power, f'{10**power:g}') for power in range(lower, upper+1)]
        position = lambda value: 545-(math.log10(value)-lower)/(upper-lower)*410
        unit = 'ms (log10); points=median, interval=Q25-Q75 (not a confidence interval)'
    else:
        maximum = 1. if metric.endswith('ssim') else math.ceil(max(r[metric] for r in rows)/5)*5
        ticks = [(maximum*i/5, f'{maximum*i/5:g}') for i in range(6)]
        position = lambda value: 545-value/maximum*410
        unit = 'SSIM Y (0..1)' if metric.endswith('ssim') else 'PSNR Y (dB)'
    pieces.append(text(65, 111, unit, 15))
    for value, label in ticks:
        y = position(value)
        pieces.append(f'<line x1="90" y1="{y:.3f}" x2="1060" y2="{y:.3f}" stroke="#dddddd"/>')
        pieces.append(text(80, y+5, label, 14, 'end'))
    pieces.append('<path d="M90 130 V545 H1060" stroke="black" fill="none"/>')
    groups = list(ALGORITHMS) if means else [f'img_{i:03d}_SRF_4' for i in range(1, 6)]
    for group_index, group in enumerate(groups):
        center = 185+group_index*192
        pieces.append(text(center, 575, group if means else group[4:7], 15, 'middle'))
        group_rows = [r for r in rows if (r['algorithm'] == group if means else r['image_id'] == group)]
        for index, row in enumerate(group_rows):
            algorithm = row['algorithm']
            if algorithm not in algorithms:
                continue
            color = COLORS[ALGORITHMS.index(algorithm)]
            offset = (algorithms.index(algorithm)-(len(algorithms)-1)/2)*30 if not means else 0
            x = center+offset
            if timing:
                value = row[scope]
                if value is None:
                    continue
                y, top, bottom = (position(value[key]) for key in ('median_ms', 'q75_ms', 'q25_ms'))
                pieces.append(f'<path d="M{x:.3f} {top:.3f} V{bottom:.3f} M{x-5:.3f} {top:.3f} H{x+5:.3f} M{x-5:.3f} {bottom:.3f} H{x+5:.3f}" stroke="{color}" stroke-width="2" fill="none"/>')
                pieces.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{color}"/>')
                flagged = ((algorithm == 'EDSR' and group.endswith('001_SRF_4'))
                           or (algorithm == 'ESRGAN' and group[4:7] in ('001', '003', '005')))
                if flagged:
                    pieces.append(text(x+6, y-9, '*', 20))
            else:
                y = position(row[metric])
                width = 78 if means else 24
                pieces.append(f'<rect x="{x-width/2:.3f}" y="{y:.3f}" width="{width}" height="{545-y:.3f}" fill="{color}"/>')
                digits = 3 if metric.endswith('ssim') else 2
                pieces.append(text(x, y-7, format(row[metric], f'.{digits}f'), 10 if not means else 14, 'middle'))
    notes = (['CPU; Bicubic: warm-up3+n30; models: warm-up3+n10. Different benchmark sessions.',
              '* Retained anomalies: EDSR001; ESRGAN001/003/005. No samples removed.',
              'Separate pipeline and network runs; do not subtract medians or infer hardware speedup.']
             if timing else ['Source: approved run metrics.json; full-precision values, rounding only for labels.',
                             'Y BT.601; crop4; data_range255; per-image arithmetic means for mean charts.'])
    for index, note in enumerate(notes):
        pieces.append(text(55, 615+index*24, note, 14))
    pieces.extend(['</g>', '</svg>'])
    return '\n'.join(pieces)+'\n'


def write_charts(rows: list[dict], directory: Path) -> None:
    """Write exactly six vector figures; no raster or image data is embedded."""
    from .comparison import summarize

    directory.mkdir()
    for metric, prefix in (('y_psnr', 'psnr_y'), ('y_ssim', 'ssim_y')):
        (directory/f'{prefix}_per_image.svg').write_text(
            chart(rows, prefix+' by image', metric=metric), encoding='utf-8', newline='\n')
        (directory/f'mean_{prefix}.svg').write_text(
            chart(summarize(rows), 'Mean '+prefix, metric=metric, means=True), encoding='utf-8', newline='\n')
    for scope, name in (('pipeline', 'pipeline_time_per_image'), ('network', 'network_time_per_image')):
        (directory/f'{name}.svg').write_text(
            chart(rows, 'CPU '+scope+' time by image', scope=scope), encoding='utf-8', newline='\n')
