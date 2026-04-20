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
from scripts.plot_run import generate_plots

args = load_config('sim')

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
results_dir = os.path.join('results', args.env, str(args.seed))
images_dir  = os.path.join(results_dir, 'images')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(images_dir,  exist_ok=True)

run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(results_dir, f'rewards_{run_id}.csv')
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'episode', 'steps', 'reward',
    'mean_cte', 'max_cte', 'std_cte', 'survival_rate',
    'obs_loss', 'kl_loss', 'reward_loss', 'actor_loss', 'value_loss',
])
csv_file.flush()
print(f'Logging rewards to {csv_path}')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

setup_device(args)

metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [],
    'test_episodes': [], 'test_rewards': [],
    'observation_loss': [], 'reward_loss': [], 'kl_loss': [],
    'actor_loss': [], 'value_loss': [],
    'episode_lengths': [], 'mean_cte': [],
}

controller = None
if args.human_override:
    from dreamer.envs.controller import EpisodeController
    controller = EpisodeController.from_keyboard()

env = Env(args.env, args.seed, args.max_episode_length,
          sim_path=args.sim_path, host=args.host, port=args.port,
          controller=controller, smooth_weight=args.smooth_weight,
          smooth_window=args.smooth_window, channels=args.channels)
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
        if controller:
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
            action[1] = args.throttle_base
            next_observation, reward, done = env.step(action)
            agent.D.append(next_observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(
            t + (0 if not metrics['steps'] else metrics['steps'][-1])
        )
        metrics['episodes'].append(s)  # s is 1-based

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
    if controller:
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
    with torch.no_grad():
        while True:
            buf_snap = agent.D.snapshot()
            observation, total_reward = env.reset(), 0
            belief = torch.zeros(1, args.belief_size, device=args.device)
            posterior_state = torch.zeros(1, args.state_size, device=args.device)
            action = torch.zeros(1, env.action_size, device=args.device)
            cte_history = []
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
                if hasattr(env, 'last_cte') and not args.human_override:
                    cte_history.append(abs(env.last_cte))
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
    cte_arr = np.array(cte_history) if cte_history else np.array([0.0])
    metrics['steps'].append(t + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    metrics['episode_lengths'].append(ep_len)
    metrics['mean_cte'].append(float(cte_arr.mean()))

    csv_writer.writerow([
        episode,
        metrics['steps'][-1],
        round(total_reward, 4),
        # driving quality
        round(float(cte_arr.mean()), 4),
        round(float(cte_arr.max()), 4),
        round(float(cte_arr.std()), 4),
        round(ep_len / args.max_episode_length, 4),
        # world model losses (mean over gradient steps this episode)
        round(float(np.mean(losses[0])), 4),
        round(float(np.mean(losses[2])), 4),
        round(float(np.mean(losses[1])), 4),
        round(float(np.mean(losses[4])), 4),
        round(float(np.mean(losses[5])), 4),
    ])
    csv_file.flush()

    # --- Evaluation ---
    if episode % args.test_interval == 0:
        agent.set_eval_mode()

        with torch.no_grad():
            observation = env.reset()
            total_rewards = 0
            belief = torch.zeros(args.test_episodes, args.belief_size, device=args.device)
            posterior_state = torch.zeros(args.test_episodes, args.state_size, device=args.device)
            action = torch.zeros(args.test_episodes, env.action_size, device=args.device)

            for t in tqdm(range(args.max_episode_length)):
                belief, posterior_state = agent.infer_state(
                    observation.to(device=args.device), action, belief, posterior_state
                )
                action = agent.select_action((belief, posterior_state), deterministic=True)
                next_observation, reward, done = env.step(action[0].cpu())
                total_rewards += reward
                observation = next_observation
                if done:
                    break

        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards)
        agent.save_reconstruction(images_dir, episode, args.episodes)
        generate_plots(csv_path, results_dir)

        agent.set_train_mode()

    print('episodes: {}, total_steps: {}, train_reward: {}'.format(
        metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]
    ))

    # --- Checkpoint ---
    if episode % args.checkpoint_interval == 0:
        agent.save_checkpoint(os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            torch.save(agent.D, os.path.join(results_dir, 'experience.pth'))

env.close()
csv_file.close()
print(f'Rewards saved to {csv_path}')
