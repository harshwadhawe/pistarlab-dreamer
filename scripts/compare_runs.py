"""
Compare two reward CSV files to verify reproducibility.

Usage:
  python scripts/compare_runs.py \
      results/donkey-generated-roads-v0/42/rewards_20260414_120000.csv \
      results/donkey-generated-roads-v0/42/rewards_20260414_130000.csv
"""

import argparse
import csv
import sys


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run1', help='First rewards CSV')
    parser.add_argument('run2', help='Second rewards CSV')
    parser.add_argument('--tol', type=float, default=0.01,
                        help='Absolute tolerance for "near-identical" check (default 0.01)')
    args = parser.parse_args()

    r1 = load_csv(args.run1)
    r2 = load_csv(args.run2)

    n = min(len(r1), len(r2))
    if len(r1) != len(r2):
        print(f'Warning: different episode counts ({len(r1)} vs {len(r2)}), comparing first {n}.')

    print(f'\n{"="*70}')
    print(f'  Reproducibility check  (tolerance = ±{args.tol})')
    print(f'  Run 1: {args.run1}')
    print(f'  Run 2: {args.run2}')
    print(f'{"="*70}')
    print(f'\n  {"ep":>4}  {"reward_1":>10}  {"reward_2":>10}  {"delta":>10}  {"ok":>4}')
    print(f'  {"-"*4}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*4}')

    mismatches = []
    for i in range(n):
        ep = int(r1[i]['episode'])
        v1 = r1[i]['reward']
        v2 = r2[i]['reward']
        delta = v2 - v1
        ok = abs(delta) <= args.tol
        if not ok:
            mismatches.append(ep)
        mark = 'YES' if ok else 'NO '
        print(f'  {ep:>4}  {v1:>10.4f}  {v2:>10.4f}  {delta:>+10.4f}  {mark}')

    print(f'\n  Episodes compared : {n}')
    print(f'  Identical (within ±{args.tol}) : {n - len(mismatches)}/{n}')

    if mismatches:
        print(f'  Mismatched episodes : {mismatches}')
        print('\n  Result: NOT fully reproducible — check seeding.')
        sys.exit(1)
    else:
        print('\n  Result: REPRODUCIBLE')


if __name__ == '__main__':
    main()
