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


def extract_modes(payload, split):
    out = {}
    if 'continuous_metrics' in payload:
        split_block = payload['continuous_metrics'].get(split, {})
        for mode in ['rollout', 'true_past']:
            out[mode] = {
                'fp_bps': metric(split_block, mode, 'fp_bps'),
                'r2': metric(split_block, mode, 'r2'),
                'teacher_forced_loss': metric(split_block, mode, 'teacher_forced_loss'),
                'elapsed_s': metric(split_block, mode, 'elapsed_s'),
            }
    else:
        test_metrics = payload.get('test_metrics', {})
        if isinstance(test_metrics, dict):
            for mode in ['rollout', 'true_past']:
                out[mode] = {
                    'fp_bps': metric(test_metrics, mode, 'fp_bps'),
                    'r2': metric(test_metrics, mode, 'r2'),
                    'teacher_forced_loss': metric(test_metrics, mode, 'teacher_forced_loss'),
                    'elapsed_s': metric(test_metrics, mode, 'elapsed_s'),
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare faithful Neuroformer eval outputs')
    parser.add_argument('--canonical-json', required=True)
    parser.add_argument('--reference-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split', default='test', choices=['valid', 'test'])
    args = parser.parse_args()

    canonical = load(Path(args.canonical_json))
    reference = load(Path(args.reference_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'split': args.split,
        'canonical': {
            'path': args.canonical_json,
            'token_stats': canonical.get('token_stats', {}).get(args.split),
            'modes': extract_modes(canonical, args.split),
        },
        'reference': {
            'path': args.reference_json,
            'token_stats': reference.get('token_stats', {}).get(args.split),
            'modes': extract_modes(reference, args.split),
        },
    }

    lines = ['# Faithful Neuroformer Compare', '', f'Split: {args.split}', '']
    lines.append('| Run | Mode | fp-bps | r2 | teacher_forced_loss | elapsed_s |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for run_name in ['canonical', 'reference']:
        for mode in ['rollout', 'true_past']:
            row = summary[run_name]['modes'].get(mode, {})
            lines.append(
                f"| {run_name} | {mode} | {row.get('fp_bps')} | {row.get('r2')} | {row.get('teacher_forced_loss')} | {row.get('elapsed_s')} |"
            )

    (output_dir / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
