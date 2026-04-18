"""
Server-side Dreamer trainer for real-world DonkeyCar.

Receives experience from the Pi5 car via ZMQ, trains the world model +
actor + critic, exports a fused TFLite model, and pushes it back to the car.

Backwards compatible with sim: accepts the same --models checkpoint format
produced by train_sim.py, and uses the identical Dreamer agent + hyperparameters.
train_sim.py (sim) is unchanged — this script is the real-world parallel.

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
from datetime import datetime

import numpy as np
import torch

from dreamer.config import add_common_args
from dreamer.agent import Dreamer
from dreamer.comms import ExperienceReceiver, ModelPublisher
from dreamer.utils import setup_device

# ---------------------------------------------------------------------------
# Args — shared defaults + server-specific additions
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Dreamer — real-world server trainer')

add_common_args(parser)

# Override defaults that differ for real-world server
parser.set_defaults(
    episodes=500,
    grayscale=False,      # RGB by default; pass --grayscale to match a grayscale car
    augment=True,         # sim-to-real augmentations on by default for real world
    expl_amount=0.0,      # TFLite runs deterministic
)

# Server-specific
parser.add_argument('--bind_ip',           type=str, default='*',
                    help='IP to bind ZMQ sockets (default: all interfaces)')
parser.add_argument('--push_interval',     type=int, default=1,
                    help='Export + push TFLite to car every N episodes')
parser.add_argument('--checkpoint_interval', type=int, default=50,
                    help='Save .pth checkpoint every N episodes')
parser.add_argument('--results-dir',       type=str, default='results/real')

args = parser.parse_args()
args.channels = 1 if args.grayscale else 3
args.observation_size = (args.channels, 64, 64)
args.smooth_weight = 0.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.makedirs(args.results_dir, exist_ok=True)
images_dir = os.path.join(args.results_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

setup_device(args)
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
    agent.load_checkpoint(args.models)
    print(f'[Server] Bootstrapped from sim checkpoint: {args.models}')

# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------
receiver  = ExperienceReceiver(bind_ip=args.bind_ip)
publisher = ModelPublisher(bind_ip=args.bind_ip)

# ---------------------------------------------------------------------------
# TFLite export (runs in litert conda env to avoid conflicts)
# ---------------------------------------------------------------------------
def export_and_publish(episode_count: int) -> None:
    """Export TFLite synchronously (blocking) then publish. Car is halted
    waiting for this — running inline ensures training always finishes
    before the car receives new weights."""
    tflite_path = os.path.join(args.results_dir, f'inference_{episode_count}.tflite')
    ckpt_path   = os.path.join(args.results_dir, f'export_weights_{episode_count}.pth')

    agent.save_inference_checkpoint(ckpt_path)

    cmd = [
        sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
        '--output', tflite_path,
        '--channels',       str(args.channels),
        '--belief-size',    str(args.belief_size),
        '--state-size',     str(args.state_size),
        '--action-size',    str(args.action_size),
        '--embedding-size', str(args.embedding_size),
        '--hidden-size',    str(args.hidden_size),
        '--throttle-base',  str(args.throttle_base),
    ]
    print(f'[Server] Exporting TFLite (episode {episode_count})...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[Server] Export FAILED:\n{result.stderr}')
        return
    print(result.stdout.strip())
    publisher.publish(tflite_path, step=agent.D.steps)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(episode_count: int) -> None:
    path = os.path.join(args.results_dir, f'models_{episode_count}.pth')
    agent.save_checkpoint(path)
    torch.save(agent.D, os.path.join(args.results_dir, 'experience.pth'))
    print(f'[Server] Checkpoint saved → {path}')


def save_reconstruction(episode_count: int) -> None:
    agent.save_reconstruction(images_dir, episode_count, args.episodes)
    print(f'[Server] Reconstruction image saved → images/ep_{episode_count}.png')


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
print(f'\n[Server] Waiting for episodes from car. '
      f'Training starts after {args.seed_episodes} seed episodes.\n')

episode_count = 0

while episode_count < args.episodes:
    print(f'[Server] Waiting for episode {episode_count + 1}/{args.episodes}...')
    ep = receiver.recv()

    # Car discarded the episode — no experience to train on; just unblock the loop.
    if ep.get('discarded', False):
        print(f'[Server] Episode {ep.get("episode_num", "?")} discarded by operator — skipping.')
        continue

    obs     = ep['obs']       # [T, C, 64, 64]  float32
    actions = ep['actions']   # [T, action_size] float32
    rewards = ep['rewards']   # [T]              float32
    dones   = ep['dones']     # [T]              bool
    T       = len(rewards)

    agent.append_episode(obs, actions, rewards, dones)

    episode_count += 1
    total_reward = float(rewards.sum())
    print(f'[Server] Episode {ep.get("episode_num", episode_count):>4d} | '
          f'steps {T:>4d} | reward {total_reward:>7.2f} | '
          f'buffer {agent.D.steps:>7d} steps')

    loss_info = None
    if episode_count > args.seed_episodes:
        grad_steps = args.collect_interval
        print(f'[Server] Training {grad_steps} gradient steps...')
        loss_info = agent.update_parameters(grad_steps)
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

    # After last seed episode: push initial weights to unblock the car.
    # Post-seed: always export+publish so the car is never left waiting.
    # Reconstruction images are saved every push_interval episodes only.
    if episode_count == args.seed_episodes:
        print('[Server] Seed phase complete — pushing initial model to car...')
        export_and_publish(episode_count)
    elif episode_count > args.seed_episodes:
        if episode_count % args.push_interval == 0:
            save_reconstruction(episode_count)
        export_and_publish(episode_count)

    if episode_count % args.checkpoint_interval == 0:
        save_checkpoint(episode_count)

# Final checkpoint
save_checkpoint(episode_count)
csv_file.close()
print(f'\n[Server] Training complete — {episode_count} episodes.')
