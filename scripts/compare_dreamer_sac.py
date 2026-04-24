"""
Compare Dreamer vs SAC training runs.

Usage:
    python scripts/compare_dreamer_sac.py \
        --dreamer results/donkey-generated-track-v0/42/<run>/rewards.csv \
        --sac     results/sac/<run>/rewards.csv \
        --out     plots/comparison.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


sns.set_theme(style='darkgrid', palette='muted', font_scale=1.1)
COLORS = {'Dreamer': '#1f77b4', 'SAC': '#ff7f0e'}
SMOOTH = 5


def load(path, label):
    df = pd.read_csv(path)
    df['label'] = label
    if 'episode_time' in df.columns:
        df['cum_time_min'] = df['episode_time'].cumsum() / 60
    return df


def plot_metric(ax, dreamer, sac, xcol, ycol, xlabel, ylabel, title):
    for df, name in [(dreamer, 'Dreamer'), (sac, 'SAC')]:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        x = df[xcol]
        y = df[ycol]
        y_smooth = y.rolling(SMOOTH, min_periods=1).mean()
        ax.plot(x, y, color=COLORS[name], alpha=0.2, linewidth=1)
        ax.plot(x, y_smooth, color=COLORS[name], linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def latest_csv(results_dir, env='donkey-generated-track-v0', seed='42'):
    dreamer_root = os.path.join(results_dir, env, seed)
    runs = sorted(os.listdir(dreamer_root))
    dreamer_csv = os.path.join(dreamer_root, runs[-1], 'rewards.csv')
    sac_root = os.path.join(results_dir, 'sac')
    sac_runs = sorted(os.listdir(sac_root))
    sac_csv = os.path.join(sac_root, sac_runs[-1], 'rewards.csv')
    return dreamer_csv, sac_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dreamer',   default=None)
    parser.add_argument('--sac',       default=None)
    parser.add_argument('--latest',    action='store_true', help='Auto-pick latest runs')
    parser.add_argument('--out',       default='plots/comparison.png')
    parser.add_argument('--threshold', type=float, default=500.0)
    args = parser.parse_args()

    if args.latest or (args.dreamer is None and args.sac is None):
        dreamer_path, sac_path = latest_csv('results')
        print(f'Dreamer: {dreamer_path}')
        print(f'SAC:     {sac_path}')
    else:
        dreamer_path, sac_path = args.dreamer, args.sac

    dreamer = load(dreamer_path, 'Dreamer')
    sac     = load(sac_path,     'SAC')

    for df in (dreamer, sac):
        if 'wall_time' in df.columns:
            df['wall_time_min'] = df['wall_time'] / 60
    has_time = 'wall_time_min' in dreamer.columns and 'wall_time_min' in sac.columns
    n_rows = 3 if has_time else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    fig.suptitle('Dreamer vs SAC — Sim Comparison', fontsize=15, fontweight='bold', y=1.01)

    plot_metric(axes[0, 0], dreamer, sac, 'episode', 'reward',
                'Episode', 'Reward', 'Reward vs Episodes')
    plot_metric(axes[0, 1], dreamer, sac, 'steps', 'reward',
                'Env Steps', 'Reward', 'Reward vs Env Steps')
    plot_metric(axes[1, 0], dreamer, sac, 'episode', 'survival_rate',
                'Episode', 'Survival Rate', 'Survival Rate vs Episodes')
    plot_metric(axes[1, 1], dreamer, sac, 'episode', 'mean_throttle',
                'Episode', 'Mean Throttle', 'Mean Throttle vs Episodes')

    if has_time:
        plot_metric(axes[2, 0], dreamer, sac, 'wall_time_min', 'reward',
                    'Time (min)', 'Reward', 'Reward vs Wall-Clock Time (incl. VAE/seed)')
        # Summary table in axes[2, 1]
        ax_tbl = axes[2, 1]
        ax_tbl.axis('off')
        rows = []
        for df, name in [(dreamer, 'Dreamer'), (sac, 'SAC')]:
            solved = df[df['reward'] >= args.threshold]
            rows.append([
                name,
                len(df),
                f"{df['reward'].max():.1f}",
                f"{df['reward'].tail(10).mean():.1f}",
                int(solved['episode'].iloc[0]) if not solved.empty else '—',
                f"{df['wall_time_min'].max():.1f}" if 'wall_time_min' in df.columns else '—',
            ])
        col_labels = ['Method', 'Episodes', 'Max\nReward', 'Mean\n(last 10)',
                      f'First ≥{args.threshold:.0f}', 'Total\nTime (min)']
        tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                           loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 2)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1f3b5c')
                cell.set_text_props(color='white', fontweight='bold')
            elif r == 1:
                cell.set_facecolor('#eaf0fb')
            else:
                cell.set_facecolor('#fff4e5')
        ax_tbl.set_title('Summary', fontweight='bold', pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Saved → {args.out}')
    print(f'Dreamer: {len(dreamer)} episodes  max={dreamer["reward"].max():.1f}')
    print(f'SAC:     {len(sac)} episodes  max={sac["reward"].max():.1f}')


if __name__ == '__main__':
    main()
