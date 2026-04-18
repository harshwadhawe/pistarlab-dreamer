"""
Server-side Dreamer trainer for real-world DonkeyCar.

Receives experience from the Pi5 via ZMQ, trains the world model +
actor + critic, exports a fused TFLite model, and pushes it back to the car.

All settings are in config.toml [real] section.

Usage:
  python train_real_server.py
"""

import csv
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

from dreamer.config import load_config
from dreamer.agent import Dreamer
from dreamer.comms import ExperienceReceiver, ModelPublisher, MODEL_HTTP_PORT
from dreamer.utils import setup_device

args = load_config('real')

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
# Comms
# ---------------------------------------------------------------------------
receiver  = ExperienceReceiver(bind_ip=args.bind_ip)
publisher = ModelPublisher(bind_ip=args.bind_ip)

# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------
def export_and_publish(episode_count: int) -> None:
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
# Checkpoint helpers
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

    if ep.get('discarded', False):
        print(f'[Server] Episode {ep.get("episode_num", "?")} discarded by operator — skipping.')
        continue

    obs     = ep['obs']
    actions = ep['actions']
    rewards = ep['rewards']
    dones   = ep['dones']
    T       = len(rewards)

    agent.append_episode(obs, actions, rewards, dones)

    episode_count += 1
    total_reward = float(rewards.sum())
    print(f'[Server] Episode {ep.get("episode_num", episode_count):>4d} | '
          f'steps {T:>4d} | reward {total_reward:>7.2f} | '
          f'buffer {agent.D.steps:>7d} steps')

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

    if episode_count == args.seed_episodes:
        print('[Server] Seed phase complete — pushing initial model to car...')
        export_and_publish(episode_count)
    elif episode_count > args.seed_episodes:
        if episode_count % args.push_interval == 0:
            save_reconstruction(episode_count)
        export_and_publish(episode_count)

    if episode_count % args.checkpoint_interval == 0:
        save_checkpoint(episode_count)

save_checkpoint(episode_count)
csv_file.close()
print(f'\n[Server] Training complete — {episode_count} episodes.')
