"""
SAC + VAE training on DonkeyCar sim — comparison baseline for Dreamer.

Key differences vs Dreamer:
  - No world model (RSSM)        no temporal memory
  - Q(z,a) Bellman backup        not imagination rollouts
  - Stateless: z = VAE.encode(obs_t) per frame, no carry-over between steps

Usage:
  conda activate donkeycar-dreamer
  python SAC/collect_data.py        # gather seed frames  (run once)
  python SAC/train_sim.py           # pre-train VAE → SAC loop
"""
import csv
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # SAC/ imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # project root

from dreamer.config import load_config
from dreamer.envs import Env
from dreamer.envs.controller import EpisodeController
from dreamer.utils import setup_device

from vae    import VAE
from policy import Actor, Critic
from agent  import SACAgent
from memory import ReplayBuffer


# ---------------------------------------------------------------------------
# Config + setup
# ---------------------------------------------------------------------------
args = load_config('sac')
setup_device(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

script_start = time.time()
_collection_time_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'collection_time.txt')
if os.path.exists(_collection_time_path):
    with open(_collection_time_path) as _f:
        script_start -= float(_f.read().strip())
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir    = os.path.join(args.results_dir, f'SAC_{timestamp}')
images_dir = os.path.join(run_dir, 'images')
os.makedirs(run_dir,    exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

csv_path = os.path.join(run_dir, 'rewards.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow([
        'episode', 'steps', 'reward', 'episode_time', 'wall_time',
        'survival_rate', 'mean_throttle',
        'critic_loss', 'actor_loss', 'alpha',
    ])
print(f'[SAC] run_dir={run_dir}  device={args.device}')


# ---------------------------------------------------------------------------
# Phase 1 — Pre-train VAE on seed frames
# ---------------------------------------------------------------------------
print(f'\n[SAC] Loading seed frames: {args.seed_data}')
if not os.path.exists(args.seed_data):
    print(f'[SAC] ERROR: seed data not found.\nRun:  python SAC/collect_data.py  first.')
    sys.exit(1)

raw        = np.load(args.seed_data)['frames']                              # (N,C,64,64) uint8
frames_all = torch.from_numpy(raw).float().to(args.device) / 255.0 - 0.5   # → [-0.5, 0.5]
print(f'[SAC] Seed frames: {frames_all.shape}')

vae     = VAE(channels=args.channels, z_dim=args.z_dim).to(args.device)
vae_opt = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)

# Pin 8 frames for consistent reconstruction tracking across epochs
pin_idx    = torch.randperm(len(frames_all))[:8]
pin_frames = frames_all[pin_idx]   # (8, C, 64, 64) — fixed for the whole run

def save_vae_reconstruction(epoch):
    with torch.no_grad():
        recon, _, _ = vae(pin_frames)
    imgs = torch.cat([pin_frames, recon], dim=0).clamp(-0.5, 0.5) + 0.5  # → [0,1]
    save_image(make_grid(imgs, nrow=8), os.path.join(images_dir, f'vae_epoch_{epoch:03d}.png'))

print(f'[SAC] Pre-training VAE — {args.vae_epochs} epochs  β={args.vae_beta}')
for epoch in range(1, args.vae_epochs + 1):
    perm      = torch.randperm(len(frames_all))
    n_batches = max(1, len(frames_all) // args.batch_size)
    ep_recon = ep_kl = 0.0

    for i in range(n_batches):
        batch = frames_all[perm[i * args.batch_size:(i + 1) * args.batch_size]]
        recon, mean, log_var = vae(batch)
        loss, recon_l, kl_l  = VAE.loss(recon, batch, mean, log_var, args.vae_beta)
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
        ep_recon += recon_l
        ep_kl    += kl_l

    if epoch % 10 == 0 or epoch == 1:
        print(f'  Epoch {epoch:3d}/{args.vae_epochs}  '
              f'recon={ep_recon/n_batches:.4f}  kl={ep_kl/n_batches:.4f}')
        save_vae_reconstruction(epoch)

torch.save(vae.state_dict(), os.path.join(run_dir, 'vae.pth'))
print(f'[SAC] VAE saved → {run_dir}/vae.pth\n')


# ---------------------------------------------------------------------------
# Phase 2 — SAC training loop
# ---------------------------------------------------------------------------
actor  = Actor(
    args.z_dim, args.action_size, args.hidden_size,
    fix_speed=args.fix_speed, throttle_base=args.throttle_base,
    throttle_min=args.throttle_min, throttle_max=args.throttle_max,
).to(args.device)

critic = Critic(args.z_dim, args.action_size, args.hidden_size).to(args.device)
replay = ReplayBuffer(args.experience_size, args.observation_size, args.action_size, args.device)
agent  = SACAgent(vae, actor, critic, args)

controller = EpisodeController.from_null()
env = Env(
    args.env, args.seed, args.max_episode_length,
    sim_path=args.sim_path, host=args.host, port=args.port,
    controller=controller,
    channels=args.channels,
    cte_left=args.cte_left, cte_right=args.cte_right,
    stuck_speed_threshold=args.stuck_speed_threshold,
    stuck_steps_limit=args.stuck_steps_limit,
    survival_bonus=args.survival_bonus,
    smooth_weight=args.smooth_weight, smooth_window=args.smooth_window,
    cte_terminate=True,
)

print(f'[SAC] Starting SAC — {args.episodes} episodes  seed={args.seed_episodes}')
total_steps = 0
best_reward = float('-inf')

for episode in tqdm(range(1, args.episodes + 1), desc='SAC'):
    obs     = env.reset()                        # (1,C,64,64) tensor
    obs_np  = obs.squeeze(0).cpu().numpy()       # (C,64,64) numpy
    done      = False
    ep_start  = time.time()
    ep_reward = 0.0
    ep_steps  = 0
    throttle_history = []
    losses    = {'critic_loss': 0.0, 'actor_loss': 0.0, 'alpha': 0.0}
    n_updates = 0

    while not done:
        # Action selection
        if episode <= args.seed_episodes:
            action = env.sample_random_action()
            action[0] = action[0].clamp(-1.0, 1.0)
            action[1] = action[1].clamp(args.throttle_min, args.throttle_max)
            if args.fix_speed:
                action[1] = args.throttle_base
        else:
            z = agent.encode(obs.to(args.device))
            action, _ = actor(z, deterministic=False)
            action = action.squeeze(0).cpu()

        next_obs, reward, done = env.step(action)
        next_obs_np = next_obs.squeeze(0).cpu().numpy()
        action_np   = action.detach().numpy()

        replay.add(obs_np, action_np, reward, done, next_obs_np)
        throttle_history.append(float(action_np[1]))

        obs     = next_obs
        obs_np  = next_obs_np
        ep_reward += reward
        ep_steps  += 1
        total_steps += 1

        # Per-step SAC update (after seed phase)
        if len(replay) >= args.batch_size and episode > args.seed_episodes:
            step_l = agent.update(replay, args.batch_size)
            for k in losses:
                losses[k] += step_l[k]
            n_updates += 1

    if n_updates:
        for k in losses:
            losses[k] /= n_updates

    # Checkpoint
    if ep_reward > best_reward:
        best_reward = ep_reward
        torch.save({
            'actor':  actor.state_dict(),
            'critic': critic.state_dict(),
            'vae':    vae.state_dict(),
        }, os.path.join(run_dir, 'best.pth'))

    # Log
    ep_time       = round(time.time() - ep_start, 2)
    survival_rate = round(ep_steps / args.max_episode_length, 4)
    mean_throttle = round(float(np.mean(throttle_history)) if throttle_history else 0.0, 4)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            episode, total_steps, round(ep_reward, 3),
            ep_time, round(time.time() - script_start, 1),
            survival_rate, mean_throttle,
            round(losses['critic_loss'], 4),
            round(losses['actor_loss'], 4),
            round(losses['alpha'], 4),
        ])

    tqdm.write(
        f'[Ep {episode:3d}] steps={ep_steps:4d}  reward={ep_reward:6.2f}  '
        f'critic={losses["critic_loss"]:.4f}  actor={losses["actor_loss"]:.4f}  '
        f'α={losses["alpha"]:.4f}'
    )

env.close()
torch.save({
    'actor':  actor.state_dict(),
    'critic': critic.state_dict(),
    'vae':    vae.state_dict(),
}, os.path.join(run_dir, 'latest.pth'))
print(f'\n[SAC] Done. Best reward: {best_reward:.2f}  Results → {run_dir}/')
