"""
Server-side Dreamer trainer for real-world DonkeyCar.

At startup, automatically cleans the Pi and pushes fresh init TFLite models
via rsync, then waits for episodes and trains.

All settings are in config.toml [real] section.

Usage:
  python train_real_server.py               # full run: init Pi then train
  python train_real_server.py --skip-init   # resume: skip Pi init, go straight to training
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

from dreamer.config import load_config
from dreamer.agent import Dreamer
from dreamer.comms import RsyncExperienceReceiver, RsyncModelPublisher
from dreamer.utils import setup_device

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--skip-init', action='store_true',
                     help='Skip Pi cleanup and init model push (use when resuming).')
_parser.add_argument('--throttle', type=float, default=None,
                     help='Override throttle_base from config.toml.')
_cli, _ = _parser.parse_known_args()

args = load_config('real')
args.skip_init = _cli.skip_init
if _cli.throttle is not None:
    args.throttle_base = _cli.throttle

a = vars(args)
print(
    f'[Config] experiment={a["experiment_name"] or "(none)"}  seed={a["seed"]}  episodes={a["episodes"]}  device=(set after)\n'
    f'         arch:   belief={a["belief_size"]} state={a["state_size"]} hidden={a["hidden_size"]} embed={a["embedding_size"]} {"grayscale" if a["grayscale"] else "RGB"}\n'
    f'         train:  batch={a["batch_size"]} chunk={a["chunk_size"]} collect={a["collect_interval"]} world_lr={a["world_lr"]}\n'
    f'         policy: fix_speed={a["fix_speed"]} throttle={a["throttle_base"]} expl={a["expl_amount"]}\n'
    f'         flags:  hflip={a["hflip"]} augment={a["augment"]} push_every={a["push_interval"]} ckpt_every={a["checkpoint_interval"]}'
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name   = f'{args.experiment_name}_{timestamp}' if args.experiment_name else timestamp
run_dir    = os.path.join(args.results_dir, run_name)
images_dir = os.path.join(run_dir, 'images')
models_dir = os.path.join(run_dir, 'models')
os.makedirs(run_dir,    exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

incoming_dir = os.path.join(run_dir, 'incoming')
os.makedirs(incoming_dir, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

setup_device(args)
print(f'[Server] Device: {args.device}  run → {run_dir}  models → {models_dir}')

csv_path = os.path.join(run_dir, 'rewards.csv')
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
receiver  = RsyncExperienceReceiver(pi_alias=args.pi_alias, local_inbox=incoming_dir)
publisher = RsyncModelPublisher(pi_alias=args.pi_alias)

# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------
def export_and_publish(episode_count: int) -> str | None:
    """Export TFLite, update models_dir/latest, serve to Pi. Returns tflite path or None."""
    label     = 'rgb' if args.channels == 3 else 'grayscale'
    ckpt_path = os.path.join(run_dir, f'inference_weights_{episode_count}.pth')
    tflite_run = os.path.join(run_dir, f'inference_{label}_{episode_count}.tflite')

    agent.save_inference_checkpoint(ckpt_path)

    cmd = [
        sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
        '--output',         tflite_run,
        '--channels',       str(args.channels),
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
    print(f'[Server] Exporting TFLite {label} (episode {episode_count})...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[Server] Export FAILED:\n{result.stderr}')
        return None
    print(result.stdout.strip())

    shutil.copy2(tflite_run, os.path.join(models_dir, f'inference_{label}_latest.tflite'))

    publisher.publish(tflite_run, step=agent.D.steps)
    return tflite_run


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(episode_count: int) -> None:
    path = os.path.join(run_dir, f'models_{episode_count}.pth')
    agent.save_checkpoint(path)
    agent.save_checkpoint(os.path.join(models_dir, 'latest.pth'))
    torch.save(agent.D, os.path.join(run_dir, 'experience.pth'))
    agent.save_reconstruction(images_dir, episode_count, args.episodes)
    print(f'[Server] Checkpoint saved → {path}')


# ---------------------------------------------------------------------------
# Pi init — clean + push init models
# ---------------------------------------------------------------------------
if args.skip_init:
    print('[Server] --skip-init: skipping Pi cleanup and init model push.')
else:
    print('[Server] Running Pi init (clean + push init models)...')
    init_cmd = [sys.executable, 'scripts/push_init_model.py']
    if args.models:
        init_cmd += ['--models', args.models]
    if _cli.throttle is not None:
        init_cmd += ['--throttle-base', str(args.throttle_base)]
    r = subprocess.run(init_cmd)
    if r.returncode != 0:
        print('[Server] Pi init failed — check SSH alias and Pi connectivity.')
        sys.exit(1)
    print('\n[Server] Pi init complete. Start train_real_pi.py on the Pi now.')
    input('[Server] Press Enter when Pi is running and ready...\n')


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
print(f'\n[Server] Waiting for episodes from car. '
      f'Training starts after {args.seed_episodes} seed episodes.\n')

episode_count = 0
best_reward   = float('-inf')

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

    if args.hflip:
        obs_flip     = obs[:, :, :, ::-1].copy()
        actions_flip = actions.copy()
        actions_flip[:, 0] = -actions_flip[:, 0]
        agent.append_episode(obs_flip, actions_flip, rewards, dones)

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
        print('[Server] Seed phase complete — pinning reconstruction sequences...')
        _pin_obs, _pin_actions, _, _pin_nonterminals = agent.D.sample(5, args.chunk_size)
        agent.pin_reconstruction_sequences(_pin_obs, _pin_actions, _pin_nonterminals)
        agent.save_reconstruction(images_dir, 0, args.episodes)
        print(f'[Server] Baseline reconstruction saved → {images_dir}/ep_000.png')
        export_and_publish(episode_count)
    elif episode_count > args.seed_episodes:
        tflite_path = export_and_publish(episode_count)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_checkpoint(os.path.join(models_dir, 'best.pth'))
            label = 'rgb' if args.channels == 3 else 'grayscale'
            if tflite_path:
                shutil.copy2(tflite_path, os.path.join(models_dir, f'inference_{label}_best.tflite'))
            print(f'[Server] New best: {best_reward:.2f} → {models_dir}/best.pth')

    if episode_count % 5 == 0 and episode_count > args.seed_episodes:
        agent.save_reconstruction(images_dir, episode_count, args.episodes)

    if episode_count % args.checkpoint_interval == 0:
        save_checkpoint(episode_count)

save_checkpoint(episode_count)
csv_file.close()
print(f'\n[Server] Training complete — {episode_count} episodes.')
