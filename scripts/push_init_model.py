"""
Export init TFLite models and push to Pi via rsync.

Does three things automatically:
  1. Cleans old .tflite + .pth from Pi models/ and resets rsync dirs
  2. Exports inference_rgb.tflite + inference_grayscale.tflite
  3. Rsyncs both to Pi — no pull_init_model.py needed on Pi

Architecture values come from config.toml [common] / [real].

Usage:
  # Random init (first ever run):
  python scripts/push_init_model.py

  # Bootstrap from a sim or real checkpoint:
  python scripts/push_init_model.py --models results/real/.../models_50.pth

  # Re-push without cleaning Pi (e.g. after a server crash):
  python scripts/push_init_model.py --skip-clean
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dreamer.config import load_config
from dreamer.models.world_model import TransitionModel, VisualObservationModel, RewardModel, VisualEncoder
from dreamer.models.policy import ActorModel, ValueModel

cfg = load_config('real')

parser = argparse.ArgumentParser()
parser.add_argument('--models',     type=str,   default=cfg.models,
                    help='Checkpoint .pth to use. Empty → random init weights.')
parser.add_argument('--skip-clean',  action='store_true',
                    help='Skip deleting old files on Pi (use when re-pushing after crash).')
parser.add_argument('--no-progress', action='store_true',
                    help='Suppress rsync transfer progress.')
parser.add_argument('--belief-size',    type=int,   default=cfg.belief_size)
parser.add_argument('--state-size',     type=int,   default=cfg.state_size)
parser.add_argument('--action-size',    type=int,   default=cfg.action_size)
parser.add_argument('--embedding-size', type=int,   default=cfg.embedding_size)
parser.add_argument('--hidden-size',    type=int,   default=cfg.hidden_size)
parser.add_argument('--throttle-base',  type=float, default=cfg.throttle_base)
parser.add_argument('--throttle-min',   type=float, default=cfg.throttle_min)
parser.add_argument('--throttle-max',   type=float, default=cfg.throttle_max)
parser.add_argument('--fix-speed',      action='store_true', default=cfg.fix_speed)
args = parser.parse_args()

PI_ALIAS   = cfg.pi_alias
PI_WORKDIR = cfg.pi_workdir

print(
    f'[Init] belief={args.belief_size} state={args.state_size} '
    f'embed={args.embedding_size} hidden={args.hidden_size} '
    f'throttle={args.throttle_base}  Pi={PI_ALIAS}:{PI_WORKDIR}'
)


# ---------------------------------------------------------------------------
# Step 1 — clean Pi
# ---------------------------------------------------------------------------

def clean_pi():
    print(f'\n[Init] Cleaning Pi ({PI_ALIAS})...')
    cmd = (
        f'rm -f {PI_WORKDIR}/models/*.tflite {PI_WORKDIR}/models/*.pth && '
        f'mkdir -p {PI_WORKDIR}/models && '
        f'rm -rf /tmp/dreamer && '
        f'mkdir -p /tmp/dreamer/outbox /tmp/dreamer/inbox'
    )
    r = subprocess.run(['ssh', PI_ALIAS, cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[Init] WARNING: Pi cleanup had issues:\n{r.stderr}')
    else:
        print('[Init] Pi cleaned: models/*.tflite, models/*.pth, /tmp/dreamer/* removed.')


# ---------------------------------------------------------------------------
# Step 2 — build checkpoint (random init or from .pth)
# ---------------------------------------------------------------------------

def _make_random_ckpt(channels: int, path: str):
    torch.save({
        'transition_model':  TransitionModel(
            args.belief_size, args.state_size, args.action_size,
            args.hidden_size, args.embedding_size,
        ).state_dict(),
        'observation_model': VisualObservationModel(
            args.belief_size, args.state_size, args.embedding_size,
            channels=channels,
        ).state_dict(),
        'reward_model':      RewardModel(
            args.belief_size, args.state_size, args.hidden_size,
        ).state_dict(),
        'encoder':           VisualEncoder(args.embedding_size, channels=channels).state_dict(),
        'actor_model':       ActorModel(
            args.action_size, args.belief_size, args.state_size, args.hidden_size,
            fix_speed=args.fix_speed, throttle_base=args.throttle_base,
            throttle_min=args.throttle_min, throttle_max=args.throttle_max,
        ).state_dict(),
        'value_model':       ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'value_model2':      ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'world_optimizer': {}, 'actor_optimizer': {}, 'value_optimizer': {},
    }, path)


# ---------------------------------------------------------------------------
# Step 3 — export TFLite
# ---------------------------------------------------------------------------

def _export_tflite(ckpt_path: str, channels: int, out_path: str):
    cmd = [
        sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
        '--output',         out_path,
        '--channels',       str(channels),
        '--belief-size',    str(args.belief_size),
        '--state-size',     str(args.state_size),
        '--action-size',    str(args.action_size),
        '--embedding-size', str(args.embedding_size),
        '--hidden-size',    str(args.hidden_size),
        '--throttle-base',  str(args.throttle_base),
        '--throttle-min',   str(args.throttle_min),
        '--throttle-max',   str(args.throttle_max),
    ]
    if args.fix_speed:
        cmd.append('--fix-speed')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[Init] Export failed (channels={channels}):\n{result.stderr}')
        sys.exit(1)
    print(result.stdout.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not args.skip_clean:
        clean_pi()

    with tempfile.TemporaryDirectory() as staging:
        for channels, label in [(3, 'rgb'), (1, 'grayscale')]:
            tflite_path = os.path.join(staging, f'inference_{label}.tflite')

            if args.models and os.path.exists(args.models):
                ckpt_path = args.models
                print(f'\n[Init] Using checkpoint: {ckpt_path}  (channels={channels})')
            else:
                ckpt_path = os.path.join(staging, f'init_{channels}ch.pth')
                print(f'\n[Init] Generating random init weights (channels={channels})...')
                _make_random_ckpt(channels, ckpt_path)

            print(f'[Init] Exporting inference_{label}.tflite...')
            _export_tflite(ckpt_path, channels, tflite_path)
            kb = os.path.getsize(tflite_path) // 1024
            print(f'[Init] Exported {kb} KB → {tflite_path}')

        # Rsync both tflite files to Pi in one call
        print(f'\n[Init] Pushing models to {PI_ALIAS}:{PI_WORKDIR}/models/ ...')
        rgb_src  = os.path.join(staging, 'inference_rgb.tflite')
        gray_src = os.path.join(staging, 'inference_grayscale.tflite')
        rsync_cmd = ['rsync', '-az', '--timeout=60']
        if not args.no_progress:
            rsync_cmd.append('--progress')
        r = subprocess.run(rsync_cmd + [rgb_src, gray_src, f'{PI_ALIAS}:{PI_WORKDIR}/models/'])
        if r.returncode != 0:
            print(f'[Init] rsync FAILED (code {r.returncode})')
            sys.exit(1)

    print(f'\n[Init] Done. Pi has inference_rgb.tflite + inference_grayscale.tflite.')
    print(f'[Init] Start train_real_pi.py on the Pi now.')


if __name__ == '__main__':
    main()
