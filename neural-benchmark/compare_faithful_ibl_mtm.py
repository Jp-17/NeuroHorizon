#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def metric(d, *keys, default=None):
    cur = d
    for key in keys:
        if cur is None or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare faithful IBL-MtM runs')
    parser.add_argument('--baseline-json', required=True)
    parser.add_argument('--control-json', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    baseline = load(Path(args.baseline_json))
    control = load(Path(args.control_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'baseline': {
            'path': args.baseline_json,
            'train_protocol': baseline.get('train_protocol', {}),
            'best_valid_fp_bps': metric(baseline, 'best_valid_metrics', 'fp_bps'),
            'formal_valid_fp_bps': metric(baseline, 'formal_valid_metrics', 'fp_bps'),
            'test_fp_bps': metric(baseline, 'test_metrics', 'fp_bps'),
        },
        'control': {
            'path': args.control_json,
            'train_protocol': control.get('train_protocol', {}),
            'best_valid_fp_bps': metric(control, 'best_valid_metrics', 'fp_bps'),
            'formal_valid_fp_bps': metric(control, 'formal_valid_metrics', 'fp_bps'),
            'test_fp_bps': metric(control, 'test_metrics', 'fp_bps'),
        },
    }
    if summary['baseline']['test_fp_bps'] is not None and summary['control']['test_fp_bps'] is not None:
        summary['delta_test_fp_bps'] = summary['control']['test_fp_bps'] - summary['baseline']['test_fp_bps']

    md = (
        '# Faithful IBL-MtM 250ms Compare\n\n'
        '| Run | train_mask_mode | best valid fp-bps | formal valid fp-bps | test fp-bps |\n'
        '|---|---:|---:|---:|---:|\n'
        f"| baseline | {summary['baseline']['train_protocol'].get('train_mask_mode')} | {summary['baseline']['best_valid_fp_bps']} | {summary['baseline']['formal_valid_fp_bps']} | {summary['baseline']['test_fp_bps']} |\n"
        f"| control | {summary['control']['train_protocol'].get('train_mask_mode')} | {summary['control']['best_valid_fp_bps']} | {summary['control']['formal_valid_fp_bps']} | {summary['control']['test_fp_bps']} |\n\n"
        f"Delta test fp-bps: {summary.get('delta_test_fp_bps')}\n"
    )
    (output_dir / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output_dir / 'comparison.md').write_text(md)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
