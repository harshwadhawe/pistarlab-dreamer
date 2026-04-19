"""
Plot ablation comparison across experiments, with mean ± std shaded bands.

Loads all rewards_*.csv files grouped by experiment name, computes mean and
std across seeds, and produces a publication-ready comparison figure.

Usage:
  python scripts/plot_ablation.py
  python scripts/plot_ablation.py --results results/donkey-generated-track-v0
  python scripts/plot_ablation.py --metric reward --out figures/ablation.png
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


METRICS = ['reward', 'obs_loss', 'kl_loss', 'reward_loss', 'actor_loss', 'value_loss']

EXPERIMENT_ORDER = ['baseline', 'no_kl_balance', 'no_symlog', 'no_return_norm']

LABELS = {
    'baseline':       'Baseline (all on)',
    'no_kl_balance':  'No KL balancing',
    'no_symlog':      'No symlog rewards',
    'no_return_norm': 'No return norm',
}


def load_results(results_root: str) -> dict[str, list[pd.DataFrame]]:
    """
    Returns {experiment_name: [df_seed1, df_seed2, df_seed3]}.
    Searches recursively under results_root for rewards_*.csv files.
    """
    pattern = os.path.join(results_root, '**', 'rewards_*.csv')
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise FileNotFoundError(f'No rewards CSVs found under {results_root}')

    groups: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(paths):
        fname = os.path.basename(path)           # rewards_baseline_seed1.csv
        stem = fname.replace('rewards_', '').replace('.csv', '')
        # stem = baseline_seed1 → experiment = baseline
        parts = stem.rsplit('_seed', 1)
        exp_name = parts[0] if len(parts) == 2 else stem
        groups.setdefault(exp_name, []).append(pd.read_csv(path))

    return groups


def plot_metric(groups: dict, metric: str, ax: plt.Axes, palette: dict) -> None:
    order = [e for e in EXPERIMENT_ORDER if e in groups] + \
            [e for e in groups if e not in EXPERIMENT_ORDER]

    for exp in order:
        dfs = groups[exp]
        # Align all seeds to the same episode index
        min_ep = min(df['episode'].max() for df in dfs)
        trimmed = [df[df['episode'] <= min_ep].set_index('episode')[metric] for df in dfs]
        stacked = pd.concat(trimmed, axis=1)

        episodes = stacked.index.values
        mean = stacked.mean(axis=1).values
        std  = stacked.std(axis=1).values

        color = palette.get(exp)
        label = LABELS.get(exp, exp)
        ax.plot(episodes, mean, label=label, color=color, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/donkey-generated-track-v0',
                        help='Root directory containing seed subdirs')
    parser.add_argument('--metrics', nargs='+', default=['reward'],
                        choices=METRICS, help='Metrics to plot')
    parser.add_argument('--out', default=None,
                        help='Output path (default: <results>/ablation.png)')
    args = parser.parse_args()

    groups = load_results(args.results)
    print(f'Found experiments: {list(groups.keys())}')
    for exp, dfs in groups.items():
        print(f'  {exp}: {len(dfs)} seed(s)')

    sns.set_theme(style='whitegrid', font_scale=1.1)
    palette_list = sns.color_palette('tab10', n_colors=len(groups))
    order = [e for e in EXPERIMENT_ORDER if e in groups] + \
            [e for e in groups if e not in EXPERIMENT_ORDER]
    palette = {exp: palette_list[i] for i, exp in enumerate(order)}

    n = len(args.metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)

    for ax, metric in zip(axes[0], args.metrics):
        plot_metric(groups, metric, ax, palette)

    # Single shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(groups),
               bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=10)

    plt.suptitle('Ablation Study — DonkeyCar Dreamer', fontsize=13, y=1.02)
    plt.tight_layout()

    out = args.out or os.path.join(args.results, 'ablation.png')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
