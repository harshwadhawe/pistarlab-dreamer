"""
Plot training metrics from a rewards CSV.

Called automatically during training (every test_interval episodes).
Also usable as a standalone script after training.

Usage:
  python scripts/plot_run.py results/donkey-generated-roads-v0/1/rewards_*.csv
  python scripts/plot_run.py results/real/rewards_20260418_120000.csv
"""

import glob
import os
import sys


def generate_plots(csv_path: str, out_dir: str) -> None:
    """Regenerate plots.html in out_dir from csv_path. Called during training."""
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.offline as ply

    df = pd.read_csv(csv_path)

    LOSS_COLS = ['obs_loss', 'kl_loss', 'reward_loss', 'actor_loss', 'value_loss']
    LOSS_COLS = [c for c in LOSS_COLS if c in df.columns]

    fig_traces = {
        'reward': [go.Scatter(x=df['episode'], y=df['reward'], name='reward', mode='lines')],
        **{c:     [go.Scatter(x=df['episode'], y=df[c],        name=c,        mode='lines')]
           for c in LOSS_COLS},
    }
    if 'mean_cte' in df.columns:
        fig_traces['mean_cte'] = [go.Scatter(x=df['episode'], y=df['mean_cte'], name='mean_cte', mode='lines')]

    plots = []
    for title, traces in fig_traces.items():
        fig = go.Figure(traces)
        fig.update_layout(title=title, xaxis_title='episode', yaxis_title=title, height=350)
        plots.append(ply.plot(fig, include_plotlyjs='cdn', output_type='div'))

    out = os.path.join(out_dir, 'plots.html')
    with open(out, 'w') as f:
        f.write('<html><body>' + ''.join(plots) + '</body></html>')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='+', help='One or more rewards CSV files (globs ok)')
    parser.add_argument('--out', default=None, help='Output HTML path (default: alongside CSV)')
    args = parser.parse_args()

    paths = []
    for p in args.csv:
        paths.extend(glob.glob(p))
    if not paths:
        sys.exit('No CSV files found.')

    import pandas as pd
    frames = [pd.read_csv(p) for p in sorted(paths)]
    import pandas as pd_inner
    df = pd_inner.concat(frames, ignore_index=True)

    out_dir = os.path.dirname(sorted(paths)[0])
    out = args.out or os.path.join(out_dir, 'plots.html')

    # Reuse generate_plots by writing a temp combined CSV
    tmp = os.path.join(out_dir, '_tmp_combined.csv')
    df.to_csv(tmp, index=False)
    generate_plots(tmp, out_dir)
    os.remove(tmp)
    if args.out:
        os.rename(os.path.join(out_dir, 'plots.html'), args.out)
        print(f'Saved → {args.out}')
    else:
        print(f'Saved → {out}')
