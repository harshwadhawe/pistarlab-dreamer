"""
SAC from pixels — sim training script.

Mirrors train_sim.py structure for fair comparison with Dreamer.
Same env, reward, episode control, collect_interval, and CSV format.

Usage:
    python train_sim_sac.py
    python train_sim_sac.py --automated
"""

import argparse
import csv
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from dreamer.config import load_config
from dreamer.envs import Env
from dreamer.models.sac import SACAgent
from dreamer.utils import setup_device

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--automated', action='store_true')
_cli, _ = _parser.parse_known_args()

args = load_config('sim')
args.automated = _cli.automated

a = vars(args)
print(
    f'[SAC] env={a["env"]}  seed={a["seed"]}  episodes={a["episodes"]}\n'
    f'      arch:  embed={a["embedding_size"]} hidden={a["hidden_size"]} {"grayscale" if a["grayscale"] else "RGB"}\n'
    f'      train: batch={a["batch_size"]} collect={a["collect_interval"]} lr={a["world_lr"]}\n'
    f'      policy: fix_speed={a["fix_speed"]} throttle={a["throttle_base"]}\n'
    f'      flags:  hflip={a["hflip"]} automated={args.automated}'
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name   = f'SAC_{args.experiment_name}_{timestamp}' if args.experiment_name else f'SAC_{timestamp}'
run_dir    = os.path.join('results', args.env, str(args.seed), run_name)
models_dir = os.path.join(run_dir, 'models')
os.makedirs(run_dir,    exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

csv_path   = os.path.join(run_dir, 'rewards.csv')
csv_file   = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'episode', 'steps', 'reward', 'episode_time',
    'survival_rate', 'mean_throttle',
    'critic_loss', 'actor_loss', 'alpha',
])
csv_file.flush()
print(f'Logging → {csv_path}')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

setup_device(args)

# ---------------------------------------------------------------------------
# Env + Agent
# ---------------------------------------------------------------------------
controller = None
if args.human_override:
    from dreamer.envs.controller import EpisodeController
    controller = EpisodeController.from_null() if args.automated else EpisodeController.from_keyboard()

env = Env(args.env, args.seed, args.max_episode_length,
          sim_path=args.sim_path, host=args.host, port=args.port,
          controller=controller, smooth_weight=args.smooth_weight,
          smooth_window=args.smooth_window, channels=args.channels,
          cte_left=args.cte_left, cte_right=args.cte_right,
          stuck_speed_threshold=args.stuck_speed_threshold,
          stuck_steps_limit=args.stuck_steps_limit,
          survival_bonus=args.survival_bonus,
          cte_terminate=args.automated)

agent = SACAgent(args, obs_shape=(args.channels, 64, 64), action_size=args.action_size)

# ---------------------------------------------------------------------------
# Seed episodes — random policy, no training
# ---------------------------------------------------------------------------
print(f'\n[SAC] Collecting {args.seed_episodes} seed episodes...')
seed_steps = []
for s in range(1, args.seed_episodes + 1):
    if controller and not args.automated:
        controller.flush()
        print(f'[SAC] Seed {s}/{args.seed_episodes}. Press → Right to start...')
        while True:
            if hasattr(env, 'brake'): env.brake()
            ev = controller.consume_event()
            if ev == controller.START: break
            time.sleep(0.05)

    obs, done = env.reset(), False
    prev_obs  = obs.squeeze(0).numpy()
    t = 0
    while not done:
        action = env.sample_random_action()
        if args.fix_speed:
            action[1] = args.throttle_base
        next_obs, reward, done = env.step(action)
        next_np = next_obs.squeeze(0).numpy()
        agent.D.push(prev_obs, action.numpy(), reward, next_np, done)
        prev_obs = next_np
        t += 1
    seed_steps.append(t)

metrics_steps    = [sum(seed_steps[:i+1]) for i in range(len(seed_steps))]
metrics_episodes = list(range(1, args.seed_episodes + 1))
best_reward      = float('-inf')

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for episode in tqdm(
    range(args.seed_episodes + 1, args.episodes + 1),
    total=args.episodes,
    initial=args.seed_episodes + 1,
):
    if hasattr(env, 'brake'):
        env.brake()

    # --- Gradient updates ---
    critic_loss, actor_loss, alpha = agent.update(args.batch_size, args.collect_interval)

    # --- Wait for operator ---
    if controller and not args.automated:
        controller.flush()
        print('[SAC] Training done. Press → Right to start next episode...')
        while True:
            if hasattr(env, 'brake'): env.brake()
            ev = controller.consume_event()
            if ev == controller.START: break
            time.sleep(0.05)

    # --- Episode collection ---
    ep_start = time.time()
    while True:
        obs, total_reward = env.reset(), 0.0
        prev_obs = obs.squeeze(0).numpy()
        throttle_history = []
        ep_buf = []
        done = False
        t = 0

        pbar = tqdm(range(args.max_episode_length), leave=False)
        for t in pbar:
            with torch.no_grad():
                action = agent.select_action(obs, deterministic=False)
            next_obs, reward, done = env.step(action[0])
            next_np = next_obs.squeeze(0).numpy()
            ep_buf.append((prev_obs, action[0].numpy(), reward, next_np, done))
            total_reward += reward
            throttle_history.append(float(action[0][1]))
            prev_obs = next_np
            obs = next_obs
            if done:
                pbar.close()
                break

        if getattr(env, 'discard_requested', False):
            print('[DISCARD] Episode erased — retrying...')
            continue

        for transition in ep_buf:
            agent.D.push(*transition)

        if args.hflip:
            for p_obs, act, rew, n_obs, dn in ep_buf:
                agent.D.push(
                    p_obs[:, :, ::-1].copy(),
                    np.array([-act[0], act[1]]),
                    rew,
                    n_obs[:, :, ::-1].copy(),
                    dn,
                )
        break

    ep_len       = t + 1
    ep_time      = time.time() - ep_start
    throttle_arr = np.array(throttle_history) if throttle_history else np.array([args.throttle_base])

    metrics_steps.append(ep_len + (metrics_steps[-1] if metrics_steps else 0))
    metrics_episodes.append(episode)

    csv_writer.writerow([
        episode,
        metrics_steps[-1],
        round(total_reward, 4),
        round(ep_time, 2),
        round(ep_len / args.max_episode_length, 4),
        round(float(throttle_arr.mean()), 4),
        round(float(critic_loss), 4),
        round(float(actor_loss), 4),
        round(float(alpha), 4),
    ])
    csv_file.flush()

    print(f'ep={episode}  steps={metrics_steps[-1]}  reward={total_reward:.1f}  '
          f'critic={critic_loss:.4f}  actor={actor_loss:.4f}  alpha={alpha:.4f}')

    if total_reward > best_reward:
        best_reward = total_reward
        agent.save_checkpoint(os.path.join(models_dir, 'best.pth'))
        print(f'[SAC] New best: {best_reward:.2f} → {models_dir}/best.pth')

    if episode % args.checkpoint_interval == 0:
        agent.save_checkpoint(os.path.join(run_dir, f'models_{episode}.pth'))
        agent.save_checkpoint(os.path.join(models_dir, 'latest.pth'))

env.close()
csv_file.close()
print(f'\n[SAC] Done. Results → {run_dir}')
