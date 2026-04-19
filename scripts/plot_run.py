"""
Plot training metrics from a rewards CSV.

Called automatically during training (every test_interval episodes).
Also usable as a standalone script after training.

Usage:
  python scripts/plot_run.py results/donkey-generated-track-v0/1/rewards_baseline_seed1.csv
  python scripts/plot_run.py results/donkey-generated-track-v0/1/rewards_*.csv
"""

import glob
import os
import sys


def generate_plots(csv_path: str, out_dir: str) -> None:
    """Save plots.png to out_dir from csv_path. Called during training."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    LOSS_COLS = ['obs_loss', 'kl_loss', 'reward_loss', 'actor_loss', 'value_loss']
    LOSS_COLS = [c for c in LOSS_COLS if c in df.columns]

    all_cols = ['reward'] + LOSS_COLS
    n = len(all_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, all_cols):
        ax.plot(df['episode'], df[col], linewidth=1.5)
        ax.set_title(col)
        ax.set_xlabel('episode')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, 'plots.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)


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

    import pandas as pd
    df = pd.concat([pd.read_csv(p) for p in sorted(paths)], ignore_index=True)

    out_dir = os.path.dirname(sorted(paths)[0])
    out = args.out or os.path.join(out_dir, 'plots.png')

    tmp = os.path.join(out_dir, '_tmp_combined.csv')
    df.to_csv(tmp, index=False)
    generate_plots(tmp, out_dir)
    os.remove(tmp)
    if args.out:
        os.rename(os.path.join(out_dir, 'plots.png'), args.out)
    print(f'Saved → {out}')
