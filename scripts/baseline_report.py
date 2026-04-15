"""
Compare one or two training runs by loading their metrics.pth files.

Usage:
  # Single run summary
  python scripts/baseline_report.py results/donkey-generated-roads-v0/1/metrics.pth

  # Compare baseline vs improved
  python scripts/baseline_report.py \
      results/donkey-generated-roads-v0/1/metrics.pth \
      results/donkey-generated-roads-v0/2/metrics.pth \
      --labels baseline improved
"""

import argparse
import sys
import numpy as np
import torch


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def smooth(values, window=5):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid').tolist()


def first_above(values, threshold):
    """Episode index where smoothed reward first exceeds threshold."""
    for i, v in enumerate(smooth(values, window=3)):
        if v > threshold:
            return i + 1
    return None


def summarise(name, m):
    episodes = m.get('episodes', [])
    rewards = m.get('train_rewards', [])
    cte = m.get('mean_cte', [])
    lengths = m.get('episode_lengths', [])
    obs_loss = [np.mean(x) for x in m.get('observation_loss', [])]
    kl_loss  = [np.mean(x) for x in m.get('kl_loss', [])]
    rew_loss = [np.mean(x) for x in m.get('reward_loss', [])]

    n = len(episodes)
    if n == 0:
        print(f'\n[{name}]  no data')
        return

    print(f'\n{"=" * 60}')
    print(f'  Run : {name}')
    print(f'  Episodes recorded : {n}')
    print(f'{"=" * 60}')

    # Reward summary
    r = np.array(rewards)
    print(f'\n  Reward')
    print(f'    first 10 ep   mean={r[:10].mean():.2f}  min={r[:10].min():.2f}  max={r[:10].max():.2f}')
    print(f'    last  10 ep   mean={r[-10:].mean():.2f}  min={r[-10:].min():.2f}  max={r[-10:].max():.2f}')
    fb = first_above(rewards, 0)
    print(f'    first ep with reward > 0 : {fb if fb else "not reached"}')

    # CTE summary
    if any(v > 0 for v in cte):
        c = np.array(cte)
        print(f'\n  Mean |CTE| per episode')
        print(f'    first 10 ep : {c[:10].mean():.3f}')
        print(f'    last  10 ep : {c[-10:].mean():.3f}')
        print(f'    overall min : {c.min():.3f}  (ep {int(c.argmin()) + 1})')

    # Episode length
    if lengths:
        ln = np.array(lengths)
        print(f'\n  Episode length (steps)')
        print(f'    first 10 ep : {ln[:10].mean():.0f}')
        print(f'    last  10 ep : {ln[-10:].mean():.0f}')

    # World model loss convergence
    if obs_loss:
        print(f'\n  World model losses (mean per episode)')
        header = f'    {"ep":>5}  {"obs_loss":>10}  {"kl_loss":>10}  {"rew_loss":>10}'
        print(header)
        milestones = sorted(set(
            [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ))
        for i in milestones:
            if i < len(obs_loss):
                kl  = kl_loss[i]  if i < len(kl_loss)  else float('nan')
                rew = rew_loss[i] if i < len(rew_loss) else float('nan')
                print(f'    {episodes[i]:>5}  {obs_loss[i]:>10.4f}  {kl:>10.4f}  {rew:>10.4f}')


def compare(m1, m2, labels):
    print(f'\n{"=" * 60}')
    print(f'  Comparison: {labels[0]}  vs  {labels[1]}')
    print(f'{"=" * 60}')

    for key, label in [('train_rewards', 'Reward'), ('mean_cte', 'Mean |CTE|')]:
        v1 = np.array(m1.get(key, []))
        v2 = np.array(m2.get(key, []))
        if v1.size == 0 or v2.size == 0:
            continue
        n = min(len(v1), len(v2))
        print(f'\n  {label} (first {n} eps)')
        print(f'    {"ep":>5}  {labels[0]:>12}  {labels[1]:>12}  {"delta":>10}')
        milestones = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        for i in milestones:
            if i < n:
                delta = v2[i] - v1[i]
                sign = '+' if delta >= 0 else ''
                print(f'    {i+1:>5}  {v1[i]:>12.3f}  {v2[i]:>12.3f}  {sign}{delta:>9.3f}')

    # Loss convergence speed: episode where obs_loss first drops below threshold
    for label, m in zip(labels, [m1, m2]):
        obs_loss = [np.mean(x) for x in m.get('observation_loss', [])]
        if obs_loss:
            initial = obs_loss[0]
            threshold = initial * 0.5
            ep = next((i + 1 for i, v in enumerate(obs_loss) if v < threshold), None)
            print(f'\n  {label}: obs_loss < 50% of initial ({initial:.4f}) at episode {ep if ep else "not reached"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics', nargs='+', help='Path(s) to metrics.pth')
    parser.add_argument('--labels', nargs='+', default=None)
    args = parser.parse_args()

    if len(args.metrics) > 2:
        print('Provide 1 or 2 metrics.pth files.')
        sys.exit(1)

    labels = args.labels or [f'run{i+1}' for i in range(len(args.metrics))]
    runs = [load(p) for p in args.metrics]

    for name, m in zip(labels, runs):
        summarise(name, m)

    if len(runs) == 2:
        compare(runs[0], runs[1], labels)


if __name__ == '__main__':
    main()
