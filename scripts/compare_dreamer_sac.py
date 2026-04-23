"""
Compare Dreamer vs SAC training runs.

Usage:
    python scripts/compare_dreamer_sac.py \
        --dreamer results/donkey-warehouse-v0/42/SIM_RGB_HFLIP_NOAUG_.../rewards.csv \
        --sac     results/donkey-warehouse-v0/42/SAC_SIM_RGB_HFLIP_NOAUG_.../rewards.csv \
        --out     results/comparison.html
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as ply


def load(path, label):
    df = pd.read_csv(path)
    df['label'] = label
    return df


def smooth(series, w=3):
    return series.rolling(w, min_periods=1).mean()


def make_fig(title, xcol, xlabel, dreamer, sac, ycol='reward'):
    traces = []
    for df, name, color in [(dreamer, 'Dreamer', '#1f77b4'), (sac, 'SAC', '#ff7f0e')]:
        if xcol not in df.columns:
            continue
        x = df[xcol]
        y = df[ycol]
        traces.append(go.Scatter(x=x, y=smooth(y), name=name,
                                 line=dict(color=color, width=2)))
        traces.append(go.Scatter(x=x, y=y, name=f'{name} (raw)',
                                 line=dict(color=color, width=1, dash='dot'),
                                 opacity=0.4))
    fig = go.Figure(traces)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title='Reward',
                      height=400, legend=dict(x=0.01, y=0.99))
    return ply.plot(fig, include_plotlyjs='cdn' if traces == traces[:2] else False,
                    output_type='div')


def summary_table(dreamer, sac, threshold=700.0):
    rows = []
    for df, name in [(dreamer, 'Dreamer'), (sac, 'SAC')]:
        solved_eps = df[df['reward'] >= threshold]
        first_solve = int(solved_eps['episode'].iloc[0]) if not solved_eps.empty else None
        rows.append({
            'Method':             name,
            'Episodes run':       len(df),
            'Max reward':         round(df['reward'].max(), 1),
            'Mean reward (last 5)': round(df['reward'].tail(5).mean(), 1),
            'First ep ≥ 700':    first_solve if first_solve else '—',
            'Total time (s)':    round(df['episode_time'].sum(), 1) if 'episode_time' in df.columns else '—',
            'Mean ep time (s)':  round(df['episode_time'].mean(), 1) if 'episode_time' in df.columns else '—',
        })

    header = list(rows[0].keys())
    table  = go.Figure(go.Table(
        header=dict(values=header, fill_color='#1f3b5c', font=dict(color='white', size=13), align='left'),
        cells=dict(
            values=[[r[h] for r in rows] for h in header],
            fill_color=[['#eaf0fb', '#fff4e5']],
            align='left', font=dict(size=12),
        ),
    ))
    table.update_layout(title='Summary', height=180)
    return ply.plot(table, include_plotlyjs=False, output_type='div')


def cumulative_time(df):
    if 'episode_time' not in df.columns:
        return None
    df = df.copy()
    df['cum_time'] = df['episode_time'].cumsum()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dreamer', required=True)
    parser.add_argument('--sac',     required=True)
    parser.add_argument('--out',     default='results/comparison.html')
    parser.add_argument('--threshold', type=float, default=700.0,
                        help='Reward threshold for "solved"')
    args = parser.parse_args()

    dreamer = load(args.dreamer, 'Dreamer')
    sac     = load(args.sac,     'SAC')

    dreamer_t = cumulative_time(dreamer)
    sac_t     = cumulative_time(sac)

    divs = ['<html><head><title>Dreamer vs SAC</title></head><body>']
    divs.append('<h2 style="font-family:sans-serif">Dreamer vs SAC — Sim Comparison</h2>')

    divs.append(summary_table(dreamer, sac, args.threshold))
    divs.append(make_fig('Reward vs Episodes', 'episode', 'Episode', dreamer, sac))
    divs.append(make_fig('Reward vs Env Steps', 'steps', 'Env Steps', dreamer, sac))

    if dreamer_t is not None and sac_t is not None:
        divs.append(make_fig('Reward vs Wall-Clock Time (s)', 'cum_time', 'Cumulative Time (s)',
                             dreamer_t, sac_t))

    divs.append('</body></html>')

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(divs))

    print(f'Saved → {args.out}')


import os
if __name__ == '__main__':
    main()
