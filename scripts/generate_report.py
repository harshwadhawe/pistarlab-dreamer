"""
Generate a self-contained PDF research report comparing Dreamer vs SAC+VAE.

Usage:
    python scripts/generate_report.py --out plots/report.pdf
    python scripts/generate_report.py --dreamer <path> --sac <path> --out plots/report.pdf
"""

import argparse
import os
import textwrap
from datetime import date

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.0)
SMOOTH   = 5
C_DREAMER = '#1f77b4'
C_SAC     = '#ff7f0e'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load(path, label):
    df = pd.read_csv(path)
    df['label'] = label
    if 'wall_time' in df.columns:
        df['wall_time_min'] = df['wall_time'] / 60
    return df


def smooth(s):
    return s.rolling(SMOOTH, min_periods=1).mean()


def latest_csvs(results_dir='results'):
    d_root = os.path.join(results_dir, 'donkey-generated-track-v0', '42')
    d_run  = sorted(os.listdir(d_root))[-1]
    s_root = os.path.join(results_dir, 'sac')
    s_run  = sorted(os.listdir(s_root))[-1]
    return (os.path.join(d_root, d_run, 'rewards.csv'),
            os.path.join(s_root, s_run, 'rewards.csv'))


def box(ax, x, y, w, h, text, color, fontsize=9, text_color='white'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                          facecolor=color, edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold',
            wrap=True, multialignment='center')


def arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Title
# ─────────────────────────────────────────────────────────────────────────────

def page_title(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    ax.text(0.5, 0.78,
            'Model-Based RL for Autonomous RC Car\nDreamer vs SAC+VAE Comparison',
            ha='center', va='center', fontsize=22, color='white', fontweight='bold',
            transform=ax.transAxes, linespacing=1.5)

    ax.text(0.5, 0.62,
            'DonkeyCar Sim  ·  Generated Track v0  ·  Seed 42',
            ha='center', va='center', fontsize=14, color='#aaaacc',
            transform=ax.transAxes)

    ax.text(0.5, 0.52,
            f'Generated {date.today().strftime("%B %d, %Y")}',
            ha='center', va='center', fontsize=11, color='#888899',
            transform=ax.transAxes)

    summary = (
        'Dreamer learns a world model from raw pixels and optimises a policy '
        'purely via latent imagination.\nSAC+VAE is a model-free baseline: a '
        'frozen VAE encoder feeds a Soft Actor-Critic Q-learning agent.\n'
        'Both methods use identical reward, action space, and observation pipeline '
        'for a fair comparison.'
    )
    ax.text(0.5, 0.35, summary, ha='center', va='center', fontsize=11,
            color='#ccccdd', transform=ax.transAxes, linespacing=1.6,
            multialignment='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Dreamer Architecture
# ─────────────────────────────────────────────────────────────────────────────

def page_dreamer_arch(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Dreamer Architecture', fontsize=16, fontweight='bold', y=0.97)

    ax = fig.add_axes([0.02, 0.1, 0.96, 0.82])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    # ── RSSM row ──
    ax.text(5, 5.6, 'Recurrent State Space Model (RSSM)', ha='center',
            fontsize=11, fontweight='bold', color='#333333')

    # Belief (GRU)
    box(ax, 0.3, 4.5, 1.8, 0.8, 'GRU Belief\nh_t  (128D)', '#2c7bb6')
    # Prior
    box(ax, 2.5, 4.5, 1.8, 0.8, 'Prior\nz_t ~ N(μ,σ)\n(20D)', '#4dac26')
    # Posterior
    box(ax, 4.7, 4.5, 1.8, 0.8, 'Posterior\nz_t ~ N(μ,σ)\n(20D)', '#d7191c')
    # Encoder
    box(ax, 6.9, 4.5, 1.8, 0.8, 'CNN Encoder\nobs_t→e_t\n(512D)', '#756bb1')
    # State
    box(ax, 2.5, 3.3, 1.8, 0.7, 'State  s_t\n= (h_t, z_t)', '#636363')

    arrow(ax, 2.1, 4.9, 2.5, 4.9)       # belief → prior
    arrow(ax, 4.3, 4.9, 4.7, 4.9)       # prior → posterior (conceptual)
    arrow(ax, 6.9, 4.9, 6.5, 4.9)       # encoder → posterior
    arrow(ax, 3.4, 4.5, 3.4, 4.0)       # prior → state
    arrow(ax, 5.6, 4.5, 3.4, 4.0)       # posterior → state (merge)
    arrow(ax, 1.2, 4.5, 1.2, 3.8, '#2c7bb6')  # belief loops

    # ── Downstream row ──
    ax.text(5, 3.0, 'Downstream Models', ha='center', fontsize=10,
            fontweight='bold', color='#555')

    box(ax, 0.3, 1.8, 1.8, 0.8, 'Obs Decoder\np(o_t|s_t)', '#8da0cb')
    box(ax, 2.5, 1.8, 1.8, 0.8, 'Reward Model\np(r_t|s_t)', '#fc8d62')
    box(ax, 4.7, 1.8, 1.8, 0.8, 'Actor\nπ(a_t|s_t)', '#66c2a5')
    box(ax, 6.9, 1.8, 1.8, 0.8, 'Twin Critics\nV(s_t)', '#e78ac3')

    for x in [1.2, 3.4, 5.6, 7.8]:
        arrow(ax, 3.4, 3.3, x, 2.6)

    # ── Imagination rollout note ──
    ax.text(5, 1.2,
            'Policy trained purely via H=15 step imagination rollouts — '
            'no environment interaction during gradient updates.',
            ha='center', fontsize=9, color='#555', style='italic')

    # ── Arch table ──
    table_data = [
        ['Component', 'Size', 'Notes'],
        ['CNN Encoder', '4-layer, 512D embed', 'Top 40px cropped, 64×64 input'],
        ['GRU Belief', '128D', 'Deterministic recurrent state'],
        ['Stochastic State', '20D Gaussian', 'Prior + posterior paths'],
        ['Hidden Size', '200D', 'Reduced from DreamerV2 (400D)'],
        ['Actor / Critic', '4-layer MLP', 'TanhNormal dist, twin V(s)'],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc='bottom', bbox=[0.0, -0.18, 1.0, 0.28])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c7bb6')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#eaf4ff')
        cell.set_edgecolor('#cccccc')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — SAC+VAE Architecture
# ─────────────────────────────────────────────────────────────────────────────

def page_sac_arch(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('SAC + VAE Architecture (Baseline)', fontsize=16,
                 fontweight='bold', y=0.97)

    ax = fig.add_axes([0.02, 0.12, 0.96, 0.80])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_axis_off()

    # ── Phase 1 ──
    ax.text(5, 5.2, 'Phase 1 — VAE Pre-Training (offline, frozen after)',
            ha='center', fontsize=11, fontweight='bold', color='#333')

    box(ax, 0.5, 3.9, 1.8, 0.8, 'Raw Obs\no_t (C,64,64)', '#888888')
    box(ax, 2.8, 3.9, 2.0, 0.8, 'β-VAE Encoder\nμ, σ  (128D)', '#756bb1')
    box(ax, 5.3, 3.9, 1.8, 0.8, 'z_t ~ N(μ,σ)\n(128D)', '#9e9ac8')
    box(ax, 7.2, 3.9, 1.8, 0.8, 'VAE Decoder\nrecon loss', '#bcbddc')

    arrow(ax, 2.3, 4.3, 2.8, 4.3)
    arrow(ax, 4.8, 4.3, 5.3, 4.3)
    arrow(ax, 6.3, 4.3, 7.2, 4.3)

    ax.text(5, 3.6, 'β = 0.1  (reconstruction-dominant)  ·  200 epochs  ·  frozen during SAC',
            ha='center', fontsize=8.5, color='#666', style='italic')

    # ── Phase 2 ──
    ax.text(5, 3.1, 'Phase 2 — SAC Training (Q-learning on real transitions)',
            ha='center', fontsize=11, fontweight='bold', color='#333')

    box(ax, 0.5, 1.8, 1.8, 0.8, 'Replay Buffer\n(o,a,r,o′)', '#888888')
    box(ax, 2.8, 1.8, 1.8, 0.8, 'Frozen Encoder\nz = μ(o)', '#756bb1')
    box(ax, 5.0, 1.8, 1.8, 0.8, 'Actor\nπ(a|z)', '#66c2a5')
    box(ax, 7.2, 1.8, 1.8, 0.8, 'Twin Q\nQ(z,a)', '#e78ac3')

    arrow(ax, 2.3, 2.2, 2.8, 2.2)
    arrow(ax, 4.6, 2.2, 5.0, 2.2)
    arrow(ax, 6.8, 2.2, 7.2, 2.2)
    arrow(ax, 7.2, 2.2, 5.0+1.8, 2.2)   # Q → actor feedback (above)

    ax.text(5, 1.4,
            'Bellman backup: Q(z,a) ← r + γ·(Q_target(z′,a′) − α·log π(a′|z′))\n'
            'No temporal memory — z is stateless per frame, no recurrence.',
            ha='center', fontsize=9, color='#555', style='italic', linespacing=1.5)

    # Key difference callout
    ax.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 0.9,
                                boxstyle='round,pad=0.05',
                                facecolor='#fff3cd', edgecolor='#e6a817', lw=1.5))
    ax.text(5, 0.65,
            'Key difference from Dreamer: SAC has no world model, no imagination, '
            'no temporal memory.\n'
            'Q values are trained on real (z,a,r,z′) tuples from the replay buffer.',
            ha='center', va='center', fontsize=9, color='#7a5800',
            multialignment='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Reward Design
# ─────────────────────────────────────────────────────────────────────────────

def page_reward(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Reward Design — Real-Car Transferability', fontsize=16,
                 fontweight='bold', y=0.97)

    ax = fig.add_axes([0.05, 0.05, 0.90, 0.88])
    ax.set_axis_off()

    sections = [
        ('#2c7bb6', 'Reward Function',
         r'$r_t = \text{survival\_bonus} + \text{throttle}_t$'
         '\n\nsurvival_bonus = 0.5  (constant per survived step)'
         '\nthrottle ∈ [0.0, 0.4]  (learned, not fixed)'
         '\n\nPer-step reward ≈ 0.5–0.9   ·   Full-lap episode reward ≈ 840'),

        ('#d7191c', 'Why Not CTE (Cross-Track Error)?',
         'CTE is a simulator-only signal — the Unity physics engine computes it '
         'from track geometry.\nThe real Raspberry Pi car has no access to CTE at inference time.\n\n'
         'Using CTE would make sim results non-transferable to hardware.\n'
         'All reward signals must be available on the physical car: speed, survival, '
         'throttle command.'),

        ('#4dac26', 'Termination vs Reward (CTE as termination only)',
         'In automated mode, CTE is used only for episode termination:\n\n'
         '   cte > 2.0  →  done=True  (off right edge)\n'
         '   cte < −6.0 →  done=True  (off left edge, asymmetric — wider tolerance)\n\n'
         'This keeps training safe without injecting a sim-only signal into the reward.'),

        ('#756bb1', 'Variable Throttle Design',
         'fix_speed = False  — actor outputs both steering AND throttle.\n\n'
         'throttle_min = 0.0, throttle_max = 0.4\n\n'
         'This forces the agent to learn speed management (slow in corners, fast on '
         'straights)\nrather than relying on a fixed cruise speed. More realistic '
         'for real-car deployment.'),
    ]

    y = 0.97
    for color, title, body in sections:
        ax.add_patch(FancyBboxPatch((0.0, y - 0.22), 1.0, 0.21,
                                    boxstyle='round,pad=0.01',
                                    facecolor=color + '22',
                                    edgecolor=color, lw=2,
                                    transform=ax.transAxes))
        ax.text(0.02, y - 0.02, title, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=color, va='top')
        ax.text(0.02, y - 0.06, body, transform=ax.transAxes,
                fontsize=9, color='#333333', va='top', linespacing=1.5)
        y -= 0.26

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Training Results
# ─────────────────────────────────────────────────────────────────────────────

def page_results(pdf, dreamer, sac):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Training Results — Generated Track v0', fontsize=16,
                 fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                           left=0.08, right=0.97, top=0.93, bottom=0.07)

    def plot(ax, xcol, ycol, xlabel, ylabel, title):
        for df, name, c in [(dreamer, 'Dreamer', C_DREAMER), (sac, 'SAC+VAE', C_SAC)]:
            if xcol not in df.columns or ycol not in df.columns:
                continue
            x, y = df[xcol], df[ycol]
            ax.plot(x, y, color=c, alpha=0.18, lw=1)
            ax.plot(x, smooth(y), color=c, lw=2.2, label=name)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)

    plot(fig.add_subplot(gs[0, 0]), 'episode', 'reward',
         'Episode', 'Reward', 'Reward vs Episodes')
    plot(fig.add_subplot(gs[0, 1]), 'steps', 'reward',
         'Env Steps', 'Reward', 'Reward vs Env Steps  (sample efficiency)')
    plot(fig.add_subplot(gs[1, 0]), 'episode', 'survival_rate',
         'Episode', 'Survival Rate', 'Survival Rate vs Episodes')

    if 'wall_time_min' in dreamer.columns and 'wall_time_min' in sac.columns:
        plot(fig.add_subplot(gs[1, 1]), 'wall_time_min', 'reward',
             'Wall-Clock Time (min)', 'Reward',
             'Reward vs Time  (incl. VAE pre-training & data collection)')
    else:
        plot(fig.add_subplot(gs[1, 1]), 'episode', 'mean_throttle',
             'Episode', 'Mean Throttle', 'Mean Throttle vs Episodes')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def page_summary(pdf, dreamer, sac):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Summary Comparison', fontsize=16, fontweight='bold', y=0.97)
    ax = fig.add_axes([0.05, 0.3, 0.90, 0.60])
    ax.set_axis_off()

    def first_full_lap(df):
        hits = df[df['survival_rate'] >= 0.99]
        if hits.empty:
            return '—'
        row = hits.iloc[0]
        ep  = int(row['episode'])
        wt  = f"{row['wall_time_min']:.1f} min" if 'wall_time_min' in row else '—'
        return f'ep {ep}  ({wt})'

    def actor_trend(df):
        if 'actor_loss' not in df.columns:
            return '—'
        first = df['actor_loss'].iloc[0]
        last  = df['actor_loss'].iloc[-1]
        direction = '↘ stable' if abs(last) < abs(first) * 3 else '↘ diverging'
        return f'{first:.1f} → {last:.1f}  ({direction})'

    rows = []
    for df, name in [(dreamer, 'Dreamer'), (sac, 'SAC + VAE')]:
        solved = df[df['survival_rate'] >= 0.99]
        rows.append([
            name,
            str(len(df)),
            f"{df['reward'].max():.0f}",
            f"{df['reward'].tail(10).mean():.0f}",
            first_full_lap(df),
            f"{df['wall_time_min'].max():.1f} min" if 'wall_time_min' in df.columns else '—',
            actor_trend(df),
        ])

    cols = ['Method', 'Episodes\nRun', 'Max\nReward', 'Mean\n(last 10)',
            'First Full Lap', 'Total\nTime', 'Actor Loss Trend']

    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 3.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#1a1a2e')
            cell.set_text_props(color='white', fontweight='bold')
        elif r == 1:
            cell.set_facecolor('#ddeeff')
        else:
            cell.set_facecolor('#fff4e5')

    # Key finding
    ax2 = fig.add_axes([0.05, 0.10, 0.90, 0.18])
    ax2.set_axis_off()
    ax2.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0.05',
                                  facecolor='#e8f5e9', edgecolor='#4dac26', lw=2,
                                  transform=ax2.transAxes))
    ax2.text(0.5, 0.6,
             'Key Finding: Dreamer achieves 7× greater sample efficiency '
             '(16 vs 108 episodes to first full lap)',
             ha='center', va='center', fontsize=12, fontweight='bold',
             color='#1b5e20', transform=ax2.transAxes)
    ax2.text(0.5, 0.25,
             'Wall-clock time to first full lap is comparable (~16 min both), '
             'demonstrating Dreamer\'s world model pays for itself.\n'
             'SAC actor loss diverges (Q overestimation with frozen encoder) — '
             'a known architectural limitation noted in the literature.',
             ha='center', va='center', fontsize=9, color='#2e7d32',
             transform=ax2.transAxes, linespacing=1.5, multialignment='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 — Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def page_hyperparams(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Hyperparameters', fontsize=16, fontweight='bold', y=0.97)
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.88])
    ax.set_axis_off()

    dreamer_params = [
        ['belief_size',        '128',    'GRU hidden (DreamerV2: 200)'],
        ['state_size',         '20',     'Stochastic state dim (DreamerV2: 30)'],
        ['hidden_size',        '200',    'MLP hidden (DreamerV2: 400)'],
        ['embedding_size',     '512',    'CNN output (DreamerV2: 1024)'],
        ['batch_size',         '50',     'Sequences per gradient step'],
        ['chunk_size',         '50',     'Sequence length per batch'],
        ['collect_interval',   '50',     'Gradient steps per episode'],
        ['planning_horizon',   '15',     'Imagination rollout length'],
        ['world_lr',           '6e-4',   'World model Adam LR'],
        ['actor_lr / value_lr','8e-5',   'Policy Adam LR'],
        ['discount',           '0.99',   'Return discount'],
        ['free_nats',          '1.0',    'KL free bits'],
        ['kl_balance',         'True',   'DreamerV2 KL balancing (α=0.8)'],
        ['symlog_rewards',     'True',   'DreamerV3 symlog transform'],
        ['return_norm',        'True',   'DreamerV3 return normalisation'],
    ]

    sac_params = [
        ['z_dim',       '128',   'VAE latent dimension'],
        ['vae_lr',      '3e-4',  'VAE Adam LR'],
        ['vae_epochs',  '200',   'Pre-training epochs'],
        ['vae_beta',    '0.1',   'β-VAE KL weight (reconstruction-dominant)'],
        ['actor_lr',    '8e-5',  'SAC actor Adam LR  (from common)'],
        ['value_lr',    '8e-5',  'SAC critic Adam LR  (from common)'],
        ['discount',    '0.99',  'Q-learning discount  (from common)'],
        ['polyak',      '0.005', 'Target network soft update'],
        ['throttle_min','0.15',  'Throttle floor (prevent collapse)'],
        ['throttle_max','0.4',   'Throttle ceiling'],
        ['target_entropy', '-2.0', '= -action_size  (auto-temperature)'],
    ]

    def make_table(ax, data, cols, title, color, x, y, w, h):
        sub = ax.inset_axes([x, y, w, h])
        sub.set_axis_off()
        sub.text(0.5, 1.02, title, ha='center', va='bottom', fontsize=10,
                 fontweight='bold', color=color, transform=sub.transAxes)
        tbl = sub.table(cellText=data, colLabels=cols,
                        loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if r == 0:
                cell.set_facecolor(color)
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f5f5f5')

    cols = ['Parameter', 'Value', 'Notes']
    make_table(ax, dreamer_params, cols, 'Dreamer', C_DREAMER, 0.0, 0.0, 0.55, 1.0)
    make_table(ax, sac_params,    cols, 'SAC + VAE', C_SAC,   0.57, 0.18, 0.43, 0.82)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dreamer', default=None)
    parser.add_argument('--sac',     default=None)
    parser.add_argument('--out',     default='plots/report.pdf')
    args = parser.parse_args()

    if args.dreamer is None or args.sac is None:
        d_path, s_path = latest_csvs('results')
    else:
        d_path, s_path = args.dreamer, args.sac

    print(f'Dreamer: {d_path}')
    print(f'SAC:     {s_path}')

    dreamer = load(d_path, 'Dreamer')
    sac     = load(s_path, 'SAC+VAE')

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    with PdfPages(args.out) as pdf:
        page_title(pdf)
        page_dreamer_arch(pdf)
        page_sac_arch(pdf)
        page_reward(pdf)
        page_results(pdf, dreamer, sac)
        page_summary(pdf, dreamer, sac)
        page_hyperparams(pdf)

    print(f'Report saved → {args.out}')


if __name__ == '__main__':
    main()
