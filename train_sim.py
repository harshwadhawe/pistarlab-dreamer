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
from dreamer.agent import Dreamer
from dreamer.utils import setup_device

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--automated', action='store_true',
                     help='Automated mode: controller-only termination, CTE logged for analysis')
_cli, _ = _parser.parse_known_args()

args = load_config('sim')
args.automated = _cli.automated

a = vars(args)
print(
    f'[Config] env={a["env"]}  seed={a["seed"]}  episodes={a["episodes"]}  device=(set after)\n'
    f'         arch:   belief={a["belief_size"]} state={a["state_size"]} hidden={a["hidden_size"]} embed={a["embedding_size"]} {"grayscale" if a["grayscale"] else "RGB"}\n'
    f'         train:  batch={a["batch_size"]} chunk={a["chunk_size"]} collect={a["collect_interval"]} world_lr={a["world_lr"]} actor_lr={a["actor_lr"]}\n'
    f'         policy: horizon={a["planning_horizon"]} discount={a["discount"]} expl={a["expl_amount"]} fix_speed={a["fix_speed"]} throttle={a["throttle_base"]}\n'
    f'         flags:  hflip={a["hflip"]} augment={a["augment"]} kl_balance={a["kl_balance"]} symlog={a["symlog_rewards"]} return_norm={a["return_norm"]}\n'
    f'         term:   cte_left={a["cte_left"]} cte_right={a["cte_right"]} stuck_spd={a["stuck_speed_threshold"]} stuck_steps={a["stuck_steps_limit"]}'
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name   = f'{args.experiment_name}_{timestamp}' if args.experiment_name else timestamp
run_dir    = os.path.join('results', args.env, str(args.seed), run_name)
images_dir = os.path.join(run_dir, 'images')
models_dir = os.path.join(run_dir, 'models')
os.makedirs(run_dir,    exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

csv_path  = os.path.join(run_dir, 'rewards.csv')
csv_file  = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'episode', 'steps', 'reward', 'episode_time',
    'mean_cte', 'max_cte', 'min_cte', 'std_cte', 'survival_rate', 'mean_throttle',
    'obs_loss', 'kl_loss', 'reward_loss', 'actor_loss', 'value_loss',
])
csv_file.flush()
print(f'Logging rewards to {csv_path}')
print(f'Models       → {models_dir}/')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

setup_device(args)

best_reward = float('-inf')

metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [],
    'observation_loss': [], 'reward_loss': [], 'kl_loss': [],
    'actor_loss': [], 'value_loss': [],
    'episode_lengths': [], 'mean_cte': [],
}

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
agent = Dreamer(args)

# ---------------------------------------------------------------------------
# Load checkpoint / seed episodes
# ---------------------------------------------------------------------------
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    agent.D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = (
        [agent.D.steps] * agent.D.episodes,
        list(range(1, agent.D.episodes + 1)),
    )
elif not args.test:
    for s in range(1, args.seed_episodes + 1):
        if controller and not args.automated:
            controller.flush()
            print(f'[Sim] Seed episode {s}/{args.seed_episodes}. Press → Right to start...')
            while True:
                if hasattr(env, 'brake'):
                    env.brake()
                ev = controller.consume_event()
                if ev == controller.START:
                    break
                time.sleep(0.05)
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            if args.fix_speed:
                action[1] = args.throttle_base
            next_observation, reward, done = env.step(action)
            agent.D.append(next_observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(
            t + (0 if not metrics['steps'] else metrics['steps'][-1])
        )
        metrics['episodes'].append(s)  # s is 1-based

# Pin 5 sequences from the seed buffer for consistent decoder tracking.
# These same frames are reconstructed at every checkpoint to show improvement.
_pin_obs, _pin_actions, _, _pin_nonterminals = agent.D.sample(5, args.chunk_size)
agent.pin_reconstruction_sequences(_pin_obs, _pin_actions, _pin_nonterminals)
agent.save_reconstruction(images_dir, 0, args.episodes)  # ep_000 = before any training
print(f'[Sim] Pinned 5 sequences for decoder tracking → {images_dir}/ep_000.png')

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for episode in tqdm(
    range(metrics['episodes'][-1] + 1, args.episodes + 1),
    total=args.episodes,
    initial=metrics['episodes'][-1] + 1,
):
    # Park the car while the world model trains (zero throttle)
    if hasattr(env, 'brake'):
        env.brake()

    # --- World model + actor + critic updates ---
    loss_info = agent.update_parameters(args.collect_interval)

    losses = tuple(zip(*loss_info))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[4])
    metrics['value_loss'].append(losses[5])

    # --- Wait for operator before starting episode ---
    if controller and not args.automated:
        controller.flush()
        print('[Sim] Training done. Press → Right to start next episode...')
        while True:
            if hasattr(env, 'brake'):
                env.brake()
            ev = controller.consume_event()
            if ev == controller.START:
                break
            time.sleep(0.05)

    # --- Data collection (retries on discard) ---
    ep_start = time.time()
    with torch.no_grad():
        while True:
            buf_snap = agent.D.snapshot()
            observation, total_reward = env.reset(), 0
            belief = torch.zeros(1, args.belief_size, device=args.device)
            posterior_state = torch.zeros(1, args.state_size, device=args.device)
            action = torch.zeros(1, env.action_size, device=args.device)
            cte_history = []
            throttle_history = []
            ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

            pbar = tqdm(range(args.max_episode_length))
            for t in pbar:
                belief, posterior_state = agent.infer_state(
                    observation.to(device=args.device), action, belief, posterior_state
                )
                action = agent.select_action((belief, posterior_state), deterministic=False)
                next_observation, reward, done = env.step(action[0].cpu())
                ep_obs.append(next_observation)
                ep_actions.append(action.cpu())
                ep_rewards.append(reward)
                ep_dones.append(done)
                total_reward += reward
                observation = next_observation
                throttle_history.append(float(action[0][1].cpu()))
                if hasattr(env, 'last_cte') and (args.automated or not args.human_override):
                    cte_history.append(env.last_cte)
                if done:
                    pbar.close()
                    break

            if getattr(env, 'discard_requested', False):
                agent.D.restore(buf_snap)
                print('[DISCARD] Episode erased — retrying...')
                continue

            # Append original episode
            for o, a, r, d in zip(ep_obs, ep_actions, ep_rewards, ep_dones):
                agent.D.append(o, a, r, d)

            # Append horizontally flipped copy (flip all frames + negate steering)
            if args.hflip:
                for o, a, r, d in zip(ep_obs, ep_actions, ep_rewards, ep_dones):
                    o_flip = torch.flip(o, [-1])
                    a_flip = a.clone()
                    a_flip[:, 0] = -a_flip[:, 0]
                    agent.D.append(o_flip, a_flip, r, d)
            break

    ep_len = t + 1
    cte_arr      = np.array(cte_history)      if cte_history      else np.array([0.0])
    throttle_arr = np.array(throttle_history) if throttle_history else np.array([args.throttle_base])
    metrics['steps'].append(t + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    metrics['episode_lengths'].append(ep_len)
    metrics['mean_cte'].append(float(np.abs(cte_arr).mean()))

    csv_writer.writerow([
        episode,
        metrics['steps'][-1],
        round(total_reward, 4),
        round(time.time() - ep_start, 2),
        # driving quality (mean=abs mean, max/min=signed for left/right visibility)
        round(float(np.abs(cte_arr).mean()), 4),
        round(float(cte_arr.max()), 4),
        round(float(cte_arr.min()), 4),
        round(float(cte_arr.std()), 4),
        round(ep_len / args.max_episode_length, 4),
        round(float(throttle_arr.mean()), 4),
        # world model losses (mean over gradient steps this episode)
        round(float(np.mean(losses[0])), 4),
        round(float(np.mean(losses[2])), 4),
        round(float(np.mean(losses[1])), 4),
        round(float(np.mean(losses[4])), 4),
        round(float(np.mean(losses[5])), 4),
    ])
    csv_file.flush()

    print('episodes: {}, total_steps: {}, train_reward: {}'.format(
        metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]
    ))

    # --- Best model ---
    if total_reward > best_reward:
        best_reward = total_reward
        agent.save_checkpoint(os.path.join(models_dir, 'best.pth'))
        print(f'[Sim] New best: {best_reward:.2f} → {models_dir}/best.pth')

    # --- Reconstruction image every 5 episodes ---
    if episode % 5 == 0:
        agent.save_reconstruction(images_dir, episode, args.episodes)

    # --- Checkpoint ---
    if episode % args.checkpoint_interval == 0:
        agent.save_checkpoint(os.path.join(run_dir, 'models_%d.pth' % episode))
        agent.save_checkpoint(os.path.join(models_dir, 'latest.pth'))
        agent.save_reconstruction(images_dir, episode, args.episodes)
        if args.checkpoint_experience:
            torch.save(agent.D, os.path.join(run_dir, 'experience.pth'))

env.close()
csv_file.close()
print(f'Rewards saved to {csv_path}')
