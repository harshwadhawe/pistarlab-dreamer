"""
Server-side Dreamer trainer for real-world DonkeyCar.

Receives experience from the Pi5 car via ZMQ, trains the world model +
actor + critic, exports a fused TFLite model, and pushes it back to the car.

Backwards compatible with sim: accepts the same --models checkpoint format
produced by train.py, and uses the identical Dreamer agent + hyperparameters.
train.py (sim) is unchanged — this script is the real-world parallel.

Setup (server, donkeycar-dreamer conda env):
  pip install pyzmq

TFLite export runs in a subprocess using the same donkeycar-dreamer conda env
(litert_torch is installed there). --litert_env defaults to donkeycar-dreamer.

Usage:
  # Bootstrap from sim checkpoint, then train on real car data:
  python server_trainer.py --models results/donkey-generated-roads-v0/1/models_500.pth

  # Train from scratch (random init):
  python server_trainer.py --episodes 500

  # Tune push interval and buffer warmup:
  python server_trainer.py --models models_500.pth --push_interval 2 --min_buffer_steps 1000
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F

from dreamer.agent import Dreamer
from dreamer.comms import ExperienceReceiver, ModelPublisher

# ---------------------------------------------------------------------------
# Args — same architecture defaults as train.py for checkpoint compatibility
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Dreamer — real-world server trainer')

# Architecture (must match the checkpoint you load)
parser.add_argument('--embedding-size',  type=int,   default=1024)
parser.add_argument('--hidden-size',     type=int,   default=300)
parser.add_argument('--belief-size',     type=int,   default=200)
parser.add_argument('--state-size',      type=int,   default=30)
parser.add_argument('--action_size',     type=int,   default=2)
parser.add_argument('--cnn-act',         type=str,   default='relu', choices=dir(F))
parser.add_argument('--dense-act',       type=str,   default='elu',  choices=dir(F))
parser.add_argument('--free-nats',       type=float, default=1.0)
parser.add_argument('--bit-depth',       type=int,   default=8)
parser.add_argument('--reward_scale',    type=int,   default=10)
parser.add_argument('--pcont',           action='store_true')
parser.add_argument('--pcont_scale',     type=int,   default=10)
parser.add_argument('--symbolic',        action='store_true')

# Training
parser.add_argument('--episodes',          type=int,   default=500)
parser.add_argument('--collect-interval',  type=int,   default=200,
                    help='Gradient steps per training round (use 200-400 for real world)')
parser.add_argument('--batch-size',        type=int,   default=50)
parser.add_argument('--chunk-size',        type=int,   default=50)
parser.add_argument('--experience-size',   type=int,   default=1000000)
parser.add_argument('--world_lr',          type=float, default=6e-4)
parser.add_argument('--actor_lr',          type=float, default=8e-5)
parser.add_argument('--value_lr',          type=float, default=8e-5)
parser.add_argument('--grad-clip-norm',    type=float, default=100.0)

# Policy
parser.add_argument('--planning-horizon',  type=int,   default=15)
parser.add_argument('--discount',          type=float, default=0.99)
parser.add_argument('--disclam',           type=float, default=0.95)
parser.add_argument('--polyak',            type=float, default=0.005)
parser.add_argument('--expl_amount',       type=float, default=0.0,
                    help='TFLite runs deterministic; keep at 0 for real world')
parser.add_argument('--with_logprob',      action='store_true')
parser.add_argument('--auto_temp',         action='store_true')
parser.add_argument('--temp',              type=float, default=0.003)
parser.add_argument('--kl_balance',        action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--symlog_rewards',    action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--return_norm',       action=argparse.BooleanOptionalAction, default=True)

# DonkeyCar action
parser.add_argument('--fix_speed',         action='store_true', default=True)
parser.add_argument('--throttle_base',     type=float, default=0.3)
parser.add_argument('--throttle_min',      type=float, default=0.1)
parser.add_argument('--throttle_max',      type=float, default=0.5)
parser.add_argument('--angle_min',         type=float, default=-1.0)
parser.add_argument('--angle_max',         type=float, default=1.0)
parser.add_argument('--grayscale',         action='store_true', default=True,
                    help='1-channel grayscale — must match car drive script --channels')
parser.add_argument('--observation_size',  default=None)
parser.add_argument('--augment',           action='store_true', default=True,
                    help='Sim-to-real augmentations during world model training')

# Server-specific
parser.add_argument('--bind_ip',           type=str,   default='*',
                    help='IP to bind ZMQ sockets (default: all interfaces)')
parser.add_argument('--seed-episodes',     type=int,   default=5,
                    help='Collect this many episodes before training starts (mirrors sim)')
parser.add_argument('--push_interval',     type=int,   default=1,
                    help='Export + push TFLite to car every N episodes (default 1 — every episode)')
parser.add_argument('--checkpoint_interval', type=int, default=50,
                    help='Save .pth checkpoint every N episodes')

# Paths / init
parser.add_argument('--models',            type=str,   default='',
                    help='Path to .pth checkpoint to bootstrap from (sim or prior real run)')
parser.add_argument('--experience-replay', type=str,   default='',
                    help='Path to saved replay buffer to resume from')
parser.add_argument('--results-dir',       type=str,   default='results/real')
parser.add_argument('--seed',              type=int,   default=42)
parser.add_argument('--disable-cuda',      action='store_true')

args = parser.parse_args()
args.channels = 1 if args.grayscale else 3
args.observation_size = (args.channels, 64, 64)
args.adam_epsilon = 1e-7
args.learning_rate_schedule = 0
args.smooth_weight = 0.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.makedirs(args.results_dir, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available() and not args.disable_cuda:
    args.device = torch.device('mps')
    torch.mps.manual_seed(args.seed)
else:
    args.device = torch.device('cpu')

print(f'[Server] Device: {args.device}')

run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(args.results_dir, f'rewards_{run_id}.csv')
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['episode', 'steps', 'reward',
                     'obs_loss', 'kl_loss', 'reward_loss',
                     'actor_loss', 'value_loss', 'buffer_steps'])
csv_file.flush()

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
agent = Dreamer(args)

if args.experience_replay and os.path.exists(args.experience_replay):
    agent.D = torch.load(args.experience_replay)
    print(f'[Server] Loaded replay buffer: {agent.D.steps} steps, '
          f'{agent.D.episodes} episodes')

if args.models and os.path.exists(args.models):
    ckpt = torch.load(args.models, map_location=args.device)
    agent.transition_model.load_state_dict(ckpt['transition_model'])
    agent.observation_model.load_state_dict(ckpt['observation_model'])
    agent.reward_model.load_state_dict(ckpt['reward_model'])
    agent.encoder.load_state_dict(ckpt['encoder'])
    agent.actor_model.load_state_dict(ckpt['actor_model'])
    agent.value_model.load_state_dict(ckpt['value_model'])
    agent.value_model2.load_state_dict(ckpt['value_model2'])
    print(f'[Server] Bootstrapped from sim checkpoint: {args.models}')

# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------
receiver  = ExperienceReceiver(bind_ip=args.bind_ip)
publisher = ModelPublisher(bind_ip=args.bind_ip)

# ---------------------------------------------------------------------------
# TFLite export (runs in litert conda env to avoid conflicts)
# ---------------------------------------------------------------------------
_export_lock = threading.Lock()


def export_and_publish(episode_count: int) -> None:
    """Export TFLite in background thread, publish when done."""

    def _run():
        with _export_lock:
            tflite_path = os.path.join(
                args.results_dir, f'inference_{episode_count}.tflite'
            )
            ckpt_path = os.path.join(
                args.results_dir, f'export_weights_{episode_count}.pth'
            )
            # Save inference weights (encoder + transition + actor only)
            torch.save({
                'encoder':          agent.encoder.cpu().state_dict(),
                'transition_model': agent.transition_model.cpu().state_dict(),
                'actor_model':      agent.actor_model.cpu().state_dict(),
            }, ckpt_path)
            agent.encoder.to(args.device)
            agent.transition_model.to(args.device)
            agent.actor_model.to(args.device)

            cmd = [
                sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
                '--output', tflite_path,
                '--channels', str(args.channels),
                '--belief-size', str(args.belief_size),
                '--state-size', str(args.state_size),
                '--action-size', str(args.action_size),
                '--embedding-size', str(args.embedding_size),
                '--hidden-size', str(args.hidden_size),
                '--throttle-base', str(args.throttle_base),
            ]
            print(f'[Server] Exporting TFLite (episode {episode_count})...')
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'[Server] Export FAILED:\n{result.stderr}')
                return
            print(result.stdout.strip())
            publisher.publish(tflite_path, step=agent.D.steps)

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(episode_count: int) -> None:
    path = os.path.join(args.results_dir, f'models_{episode_count}.pth')
    torch.save({
        'transition_model':  agent.transition_model.state_dict(),
        'observation_model': agent.observation_model.state_dict(),
        'reward_model':      agent.reward_model.state_dict(),
        'encoder':           agent.encoder.state_dict(),
        'actor_model':       agent.actor_model.state_dict(),
        'value_model':       agent.value_model.state_dict(),
        'value_model2':      agent.value_model2.state_dict(),
        'world_optimizer':   agent.world_optimizer.state_dict(),
        'actor_optimizer':   agent.actor_optimizer.state_dict(),
        'value_optimizer':   agent.value_optimizer.state_dict(),
    }, path)
    torch.save(agent.D, os.path.join(args.results_dir, 'experience.pth'))
    print(f'[Server] Checkpoint saved → {path}')


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
print(f'\n[Server] Waiting for episodes from car. '
      f'Training starts after {args.seed_episodes} seed episodes.\n')

episode_count = 0

while episode_count < args.episodes:
    print(f'[Server] Waiting for episode {episode_count + 1}/{args.episodes}...')
    ep = receiver.recv()

    obs     = ep['obs']       # [T, C, 64, 64]  float32
    actions = ep['actions']   # [T, action_size] float32
    rewards = ep['rewards']   # [T]              float32
    dones   = ep['dones']     # [T]              bool
    T       = len(rewards)

    # Append to replay buffer — obs already in [-0.5, 0.5]
    for t in range(T):
        agent.D.append(
            torch.as_tensor(obs[t]),
            actions[t],
            float(rewards[t]),
            bool(dones[t]),
        )

    episode_count += 1
    total_reward = float(rewards.sum())
    print(f'[Server] Episode {ep.get("episode_num", episode_count):>4d} | '
          f'steps {T:>4d} | reward {total_reward:>7.2f} | '
          f'buffer {agent.D.steps:>7d} steps')

    loss_info = None
    if episode_count > args.seed_episodes:
        print(f'[Server] Training {args.collect_interval} gradient steps...')
        loss_info = agent.update_parameters(args.collect_interval)
        losses = np.mean(loss_info, axis=0)
        obs_l, rew_l, kl_l, _, act_l, val_l = losses
        print(f'[Server] obs={obs_l:.4f} rew={rew_l:.4f} kl={kl_l:.4f} '
              f'actor={act_l:.4f} value={val_l:.4f}')
    else:
        print(f'[Server] Seed episode {episode_count}/{args.seed_episodes} — skipping training.')
        losses = [0, 0, 0, 0, 0, 0]

    csv_writer.writerow([
        episode_count, T, total_reward,
        losses[0], losses[2], losses[1], losses[4], losses[5],
        agent.D.steps,
    ])
    csv_file.flush()

    # Always push — car blocks waiting for this to unblock before next episode.
    # Background thread so training loop isn't delayed by export subprocess.
    if episode_count % args.push_interval == 0:
        export_and_publish(episode_count)

    if episode_count % args.checkpoint_interval == 0:
        save_checkpoint(episode_count)

# Final checkpoint
save_checkpoint(episode_count)
csv_file.close()
print(f'\n[Server] Training complete — {episode_count} episodes.')
