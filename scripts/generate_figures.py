"""
Generate standalone figures for the LaTeX paper.
Saves to paper/figures/.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --dreamer <path> --sac <path>
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'font.family':     'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SMOOTH     = 5
C_DREAMER  = '#2166ac'
C_SAC      = '#d6604d'
FIGDIR     = 'paper/figures'

os.makedirs(FIGDIR, exist_ok=True)


def load(path, label):
    df = pd.read_csv(path)
    df['label'] = label
    if 'wall_time' in df.columns:
        df['wall_time_min'] = df['wall_time'] / 60
    return df


def smooth(s, w=SMOOTH):
    return s.rolling(w, min_periods=1).mean()


def latest_csvs(results_dir='results'):
    d_root = os.path.join(results_dir, 'donkey-generated-track-v0', '42')
    d_run  = sorted(os.listdir(d_root))[-1]
    s_root = os.path.join(results_dir, 'sac')
    s_run  = sorted(os.listdir(s_root))[-1]
    return (os.path.join(d_root, d_run, 'rewards.csv'),
            os.path.join(s_root, s_run, 'rewards.csv'))


# ── Figure 1: training curves ─────────────────────────────────────────────────

def fig_training_curves(dreamer, sac):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    panels = [
        ('episode',       'reward',        'Episode',           'Episode Reward',    'Reward vs.\ Episodes'),
        ('steps',         'reward',        'Environment Steps', 'Episode Reward',    'Reward vs.\ Env.\ Steps'),
        ('wall_time_min', 'reward',        'Wall-Clock Time (min)', 'Episode Reward','Reward vs.\ Wall-Clock Time'),
    ]

    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, panels):
        for df, name, c in [(dreamer, 'Dreamer', C_DREAMER), (sac, 'SAC+VAE', C_SAC)]:
            if xcol not in df.columns or ycol not in df.columns:
                continue
            x, y = df[xcol], df[ycol]
            ax.plot(x, y,        color=c, alpha=0.15, lw=0.8)
            ax.plot(x, smooth(y), color=c, lw=2.0, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = os.path.join(FIGDIR, 'training_curves.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out}')


# ── Figure 2: survival rate ───────────────────────────────────────────────────

def fig_survival(dreamer, sac):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    for df, name, c in [(dreamer, 'Dreamer', C_DREAMER), (sac, 'SAC+VAE', C_SAC)]:
        x, y = df['episode'], df['survival_rate']
        ax.plot(x, y,         color=c, alpha=0.15, lw=0.8)
        ax.plot(x, smooth(y), color=c, lw=2.0, label=name)

    ax.axhline(1.0, color='gray', ls='--', lw=1, label='Full lap')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Survival Rate', fontsize=10)
    ax.set_title('Survival Rate vs.\ Episodes', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'survival_rate.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out}')


# ── Shared drawing helpers ────────────────────────────────────────────────────

def _draw_box(ax, cx, cy, w, h, lines, fc, fs=11):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0.12',
                       facecolor=fc, edgecolor='#333333', linewidth=1.6, zorder=2)
    ax.add_patch(r)
    ax.text(cx, cy, '\n'.join(lines), ha='center', va='center',
            fontsize=fs, color='white', fontweight='bold',
            multialignment='center', zorder=3, linespacing=1.45)


def _harrow(ax, x_from, x_to, y, w, label='', c='#333333'):
    ax.annotate('', xy=(x_to - w/2, y), xytext=(x_from + w/2, y),
                arrowprops=dict(arrowstyle='->', color=c, lw=2.0, mutation_scale=18),
                zorder=1)
    if label:
        ax.text((x_from + x_to) / 2, y + 0.22, label,
                ha='center', fontsize=10, color=c)


# ── Figure 3a: Dreamer world model ───────────────────────────────────────────

def fig_dreamer_rssm():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    W, H = 2.2, 0.9
    xs   = [1.3, 3.9, 6.5, 9.1, 11.7]
    y    = 2.5

    # ── Main row: Image → CNN → GRU → Posterior → State ───────────
    _draw_box(ax, xs[0], y, W, H, ['Image  $o_t$', '64×64  RGB'], '#666666')
    _draw_box(ax, xs[1], y, W, H, ['CNN Encoder', r'$e_t$  (512-D)'], '#4393c3')
    _draw_box(ax, xs[2], y, W, H, ['GRU Belief', r'$h_t$  (128-D)'], '#2166ac')
    _draw_box(ax, xs[3], y, W, H, ['Posterior', r'$z_t \!\sim\! q(z\!\mid\!h_t,e_t)$'], '#d6604d')
    _draw_box(ax, xs[4], y, W, H, ['State  $s_t$', r'$(h_t,\;z_t)$  148-D'], '#555555')

    for i in range(4):
        _harrow(ax, xs[i], xs[i+1], y, W)
    ax.text((xs[1]+xs[2])/2, y+0.24, r'$e_t$', ha='center', fontsize=10, color='#444')
    ax.text((xs[2]+xs[3])/2, y+0.24, r'$h_t$', ha='center', fontsize=10, color='#444')

    # ── Prior box above GRU ────────────────────────────────────────
    prior_y = 3.9
    _draw_box(ax, xs[2], prior_y, W, H, ['Prior', r'$z_t \!\sim\! p(z\!\mid\!h_t)$'], '#4dac26')
    ax.annotate('', xy=(xs[2], prior_y - H/2), xytext=(xs[2], y + H/2),
                arrowprops=dict(arrowstyle='->', color='#4dac26', lw=1.8,
                                mutation_scale=15), zorder=1)
    ax.text(xs[2]+0.35, (y + H/2 + prior_y - H/2)/2, r'$h_t$',
            fontsize=9.5, color='#4dac26', va='center')
    ax.text((xs[2]+xs[3])/2 + 0.8, y + 0.72,
            r'$D_{KL}[q\|p]$',
            ha='center', fontsize=9.5, color='#888', style='italic')

    # ── Recurrence: State → GRU (arc below main row) ──────────────
    ax.annotate('', xy=(xs[2] + W/2, y - 0.28),
                    xytext=(xs[4] + W/2, y - 0.28),
                arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.7,
                                connectionstyle='arc3,rad=0.30',
                                mutation_scale=15), zorder=1)
    ax.text((xs[2]+xs[4])/2 + 0.5, y - 1.05,
            r'recurrence: $h_{t-1},\;a_{t-1}$',
            ha='center', fontsize=10, color='#2166ac', style='italic')

    ax.text(6.5, 4.75, 'Dreamer — World Model (RSSM)',
            ha='center', fontsize=13, fontweight='bold', color='#222')

    out = os.path.join(FIGDIR, 'dreamer_rssm.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  {out}')


# ── Figure 3b: Dreamer policy ─────────────────────────────────────────────────

def fig_dreamer_policy():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    W, H = 2.2, 0.9

    _draw_box(ax, 1.4, 2.5, W, H, ['State  $s_t$', r'$(h_t,\;z_t)$  148-D'], '#555555')

    imag_cx, imag_W = 5.0, 3.4
    ax.add_patch(FancyBboxPatch((imag_cx - imag_W/2, 2.5 - H/2), imag_W, H,
                                boxstyle='round,pad=0.12',
                                facecolor='#8073ac', edgecolor='#333', linewidth=1.6, zorder=2))
    ax.text(imag_cx, 2.5, 'Imagination Rollout\n$H = 15$ steps  (no env. calls)',
            ha='center', va='center', fontsize=11, color='white',
            fontweight='bold', multialignment='center', zorder=3, linespacing=1.45)

    ax.annotate('', xy=(imag_cx - imag_W/2, 2.5), xytext=(1.4 + W/2, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.0,
                                mutation_scale=18), zorder=1)

    imag_right = imag_cx + imag_W/2
    _draw_box(ax, 9.0, 3.5, W, H, [r'Actor  $\pi(a \mid s_t)$', '4-layer MLP'], '#66c2a5')
    _draw_box(ax, 9.0, 1.5, W, H, ['Twin Critics  $V(s_t)$', '+ target networks'], '#e78ac3')

    ax.annotate('', xy=(9.0 - W/2, 3.5), xytext=(imag_right, 2.65),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8,
                                mutation_scale=15), zorder=1)
    ax.annotate('', xy=(9.0 - W/2, 1.5), xytext=(imag_right, 2.35),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8,
                                mutation_scale=15), zorder=1)

    ax.text(5.5, 0.6,
            r'Losses over imagined trajectories: REINFORCE (actor),  $\lambda$-return (critic)',
            ha='center', fontsize=10.5, color='#444', style='italic')
    ax.text(5.5, 4.75, 'Dreamer — Policy: Latent Actor-Critic',
            ha='center', fontsize=13, fontweight='bold', color='#222')

    out = os.path.join(FIGDIR, 'dreamer_policy.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  {out}')


# ── Figure 4a: SAC+VAE Phase 1 (β-VAE pre-training) ──────────────────────────

def fig_sac_vae():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    W, H = 2.2, 0.9
    y = 3.3

    _draw_box(ax, 1.3, y, W, H, ['Image  $o_t$', '64×64  RGB'], '#666666')
    _draw_box(ax, 4.2, y, W, H, [r'$\beta$-VAE Encoder', r'$\mu,\sigma \to z_t$  (128-D)'], '#756bb1')
    _draw_box(ax, 7.1, y, W, H, ['Latent Code', r'$z_t$  (128-D)'], '#54278f')

    _harrow(ax, 1.3, 4.2, y, W)
    _harrow(ax, 4.2, 7.1, y, W)

    dec_y = 1.7
    _draw_box(ax, 7.1, dec_y, W, H, ['VAE Decoder', r'$\hat{o}_t$'], '#9e9ac8')
    ax.annotate('', xy=(7.1, dec_y + H/2), xytext=(7.1, y - H/2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.0,
                                mutation_scale=18), zorder=1)

    ax.text(5.0, 0.72,
            r'$\mathcal{L}_{\mathrm{VAE}} = \|o_t - \hat{o}_t\|^2 \;+\; \beta\,D_{KL}(q\|p),'
            r'\quad \beta = 0.1$',
            ha='center', fontsize=11, color='#444', style='italic')
    ax.text(5.0, 4.3,
            'Pre-trained on 30 episodes of random exploration.\n'
            'Encoder weights frozen before SAC training begins.',
            ha='center', fontsize=10, color='#666', style='italic')
    ax.text(5.0, 4.8, r'SAC + $\beta$-VAE  —  Phase 1: Visual Representation',
            ha='center', fontsize=13, fontweight='bold', color='#222')

    out = os.path.join(FIGDIR, 'sac_vae.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  {out}')


# ── Figure 4b: SAC+VAE Phase 2 (Q-learning) ──────────────────────────────────

def fig_sac_rl():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    W, H = 2.2, 0.9
    mid_y = 2.5

    _draw_box(ax, 1.3, mid_y, W, H, ['Image  $o_t$', '64×64  RGB'], '#666666')
    _draw_box(ax, 4.2, mid_y, W, H, ['Frozen Encoder', r'$z_t$  (128-D)'], '#756bb1')
    _harrow(ax, 1.3, 4.2, mid_y, W)

    enc_right = 4.2 + W/2
    actor_y, q_y = 3.6, 1.4

    _draw_box(ax, 7.5, actor_y, W, H, [r'Actor  $\pi(a \mid z_t)$', '3-layer MLP'], '#66c2a5')
    _draw_box(ax, 7.5, q_y,    W, H, [r'Twin Q  $Q(z_t, a_t)$',    '3-layer MLP'], '#e78ac3')

    ax.annotate('', xy=(7.5 - W/2, actor_y), xytext=(enc_right, mid_y + 0.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8,
                                mutation_scale=15), zorder=1)
    ax.annotate('', xy=(7.5 - W/2, q_y), xytext=(enc_right, mid_y - 0.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8,
                                mutation_scale=15), zorder=1)

    _draw_box(ax, 10.5, q_y, W, H, ['Target Q-nets', '(EMA update)'], '#c0392b')
    _harrow(ax, 7.5, 10.5, q_y, W)

    ax.text(7.5, 0.55,
            r'$Q \leftarrow r + \gamma\,(\min Q^{\prime} - \alpha\log\pi)$'
            r'$\qquad \mathcal{L}_\pi = \mathbb{E}[\,\alpha\log\pi - \min Q\,]$',
            ha='center', fontsize=11, color='#444', style='italic')
    ax.text(5.0, 4.3,
            'Encoder frozen throughout — no gradient through visual backbone.\n'
            'No temporal model: each frame processed independently.',
            ha='center', fontsize=10, color='#666', style='italic')
    ax.text(6.0, 4.8, r'SAC + $\beta$-VAE  —  Phase 2: Off-Policy Q-Learning',
            ha='center', fontsize=13, fontweight='bold', color='#222')

    out = os.path.join(FIGDIR, 'sac_rl.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  {out}')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dreamer', default=None)
    parser.add_argument('--sac',     default=None)
    args = parser.parse_args()

    if args.dreamer is None or args.sac is None:
        d_path, s_path = latest_csvs('results')
    else:
        d_path, s_path = args.dreamer, args.sac

    print(f'Dreamer : {d_path}')
    print(f'SAC     : {s_path}')
    print(f'Figures → {FIGDIR}/')

    dreamer = load(d_path, 'Dreamer')
    sac     = load(s_path, 'SAC+VAE')

    fig_dreamer_rssm()
    fig_dreamer_policy()
    fig_sac_vae()
    fig_sac_rl()
    fig_training_curves(dreamer, sac)
    fig_survival(dreamer, sac)
    print('Done.')


if __name__ == '__main__':
    main()
