"""
Plot training metrics from a rewards CSV.

Called automatically during training (every test_interval episodes).
Also usable as a standalone script after training.

Usage:
  python scripts/plot_run.py results/donkey-generated-roads-v0/1/<run>/rewards.csv
  python scripts/plot_run.py results/real/<run>/rewards.csv
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style='darkgrid', font_scale=1.05)
SMOOTH = 5


def generate_plots(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)

    metric_groups = [
        ('reward',       ['reward'],                              'Reward'),
        ('losses',       [c for c in ['obs_loss', 'kl_loss', 'reward_loss'] if c in df.columns], 'World Model Loss'),
        ('policy',       [c for c in ['actor_loss', 'value_loss'] if c in df.columns],           'Policy Loss'),
        ('tracking',     [c for c in ['mean_cte', 'std_cte'] if c in df.columns],                'CTE'),
        ('behaviour',    [c for c in ['survival_rate', 'mean_throttle'] if c in df.columns],     'Behaviour'),
    ]
    metric_groups = [(k, cols, title) for k, cols, title in metric_groups if cols]

    n = len(metric_groups)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (_, cols, title) in zip(axes, metric_groups):
        for col in cols:
            y = df[col]
            y_s = y.rolling(SMOOTH, min_periods=1).mean()
            ax.plot(df['episode'], y,   alpha=0.2, linewidth=1)
            ax.plot(df['episode'], y_s, linewidth=2, label=col)
        ax.set_ylabel(title)
        ax.legend(loc='upper left')

    axes[-1].set_xlabel('Episode')
    fig.suptitle(os.path.basename(os.path.dirname(csv_path)), fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(out_dir, 'plots.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='+', help='One or more rewards CSV files (globs ok)')
    parser.add_argument('--out', default=None, help='Output PNG path (default: alongside CSV)')
    args = parser.parse_args()

    paths = []
    for p in args.csv:
        paths.extend(glob.glob(p))
    if not paths:
        sys.exit('No CSV files found.')

    frames = [pd.read_csv(p) for p in sorted(paths)]
    df = pd.concat(frames, ignore_index=True)

    out_dir = os.path.dirname(os.path.abspath(sorted(paths)[0]))
    tmp = os.path.join(out_dir, '_tmp_combined.csv')
    df.to_csv(tmp, index=False)
    out = generate_plots(tmp, out_dir)
    os.remove(tmp)

    if args.out:
        os.rename(out, args.out)
        print(f'Saved → {args.out}')
    else:
        print(f'Saved → {out}')
