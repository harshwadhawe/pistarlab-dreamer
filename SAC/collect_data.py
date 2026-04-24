"""
Collect seed frames from the DonkeyCar sim for VAE pre-training.
Drives with a random policy and saves all observations to disk.

Usage:
  conda activate donkeycar-dreamer
  python SAC/collect_data.py [--episodes 10] [--out SAC/data/frames.npz]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dreamer.config import load_config
from dreamer.envs import Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of random episodes to collect')
    parser.add_argument('--out', default='SAC/data/frames.npz',
                        help='Output path for saved frames')
    cli = parser.parse_args()

    args = load_config('sac')

    env = Env(
        args.env, args.seed, args.max_episode_length,
        sim_path=args.sim_path, host=args.host, port=args.port,
        channels=args.channels,
        cte_left=args.cte_left, cte_right=args.cte_right,
        stuck_speed_threshold=args.stuck_speed_threshold,
        stuck_steps_limit=args.stuck_steps_limit,
        survival_bonus=args.survival_bonus,
        cte_terminate=False,
    )

    all_frames = []

    for ep in range(cli.episodes):
        obs  = env.reset()   # tensor (1,C,64,64)
        done = False
        count = 0

        while not done:
            action = env.sample_random_action()
            if args.fix_speed:
                action[1] = args.throttle_base
            obs, _, done = env.step(action)
            all_frames.append(obs.squeeze(0).cpu().numpy())   # (C,64,64)
            count += 1

        print(f'[Collect] Episode {ep+1}/{cli.episodes}  frames={count}  total={len(all_frames)}')

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    arr   = np.stack(all_frames)                                         # (N,C,64,64) float32
    uint8 = ((arr + 0.5) * 255).clip(0, 255).astype(np.uint8)
    np.savez_compressed(cli.out, frames=uint8)
    kb = os.path.getsize(cli.out) // 1024
    print(f'[Collect] Saved {len(all_frames)} frames {arr.shape} → {cli.out}  ({kb} KB)')


if __name__ == '__main__':
    main()
