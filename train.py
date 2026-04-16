import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dreamer.envs import GYM_ENVS, CONTROL_SUITE_ENVS, DONKEY_CAR_ENVS, Env, EnvBatcher
from dreamer.agent import Dreamer
from dreamer.utils import lineplot, write_video

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Dreamer')

# Environment
parser.add_argument('--env', type=str, default='donkey-generated-track-v0',
                    choices=GYM_ENVS + CONTROL_SUITE_ENVS + DONKEY_CAR_ENVS)
parser.add_argument('--symbolic', action='store_true', help='Symbolic (non-image) observations')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max-episode-length', type=int, default=1000)
parser.add_argument('--action-repeat', type=int, default=1)

# World model
parser.add_argument('--embedding-size', type=int, default=1024)
parser.add_argument('--hidden-size', type=int, default=300)
parser.add_argument('--belief-size', type=int, default=200)
parser.add_argument('--state-size', type=int, default=30)
parser.add_argument('--cnn-act', type=str, default='relu', choices=dir(F))
parser.add_argument('--dense-act', type=str, default='elu', choices=dir(F))
parser.add_argument('--free-nats', type=float, default=1)  # lowered from 3 — pairs with KL balancing
parser.add_argument('--bit-depth', type=int, default=8)
parser.add_argument('--reward_scale', type=int, default=10)
parser.add_argument('--pcont', action='store_true')
parser.add_argument('--pcont_scale', type=int, default=10)

# Training
parser.add_argument('--episodes', type=int, default=30)  # use 1000+ for real training
parser.add_argument('--seed-episodes', type=int, default=5)
parser.add_argument('--collect-interval', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--chunk-size', type=int, default=50)
parser.add_argument('--experience-size', type=int, default=1000000)
parser.add_argument('--world_lr', type=float, default=6e-4)
parser.add_argument('--actor_lr', type=float, default=8e-5)
parser.add_argument('--value_lr', type=float, default=8e-5)
parser.add_argument('--adam-epsilon', type=float, default=1e-7)
parser.add_argument('--grad-clip-norm', type=float, default=100.0)
parser.add_argument('--learning-rate-schedule', type=int, default=0)

# Policy / planning
parser.add_argument('--planning-horizon', type=int, default=15)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--disclam', type=float, default=0.95)
parser.add_argument('--polyak', type=float, default=0.005,
                    help='Soft target update rate: θ_target = (1-polyak)*θ_target + polyak*θ_online')
parser.add_argument('--expl_amount', type=float, default=0.15)
parser.add_argument('--with_logprob', action='store_true')
parser.add_argument('--auto_temp', action='store_true')
parser.add_argument('--temp', type=float, default=0.003)

# Dreamer v2/v3 improvements (all on by default; use --no-X to ablate)
parser.add_argument('--kl_balance', action=argparse.BooleanOptionalAction, default=True,
                    help='KL balancing: separate dynamics/representation loss (Dreamer v2)')
parser.add_argument('--symlog_rewards', action=argparse.BooleanOptionalAction, default=True,
                    help='Symlog-compress rewards before reward model training (Dreamer v3)')
parser.add_argument('--return_norm', action=argparse.BooleanOptionalAction, default=True,
                    help='Normalise returns by running 5th/95th percentile range (Dreamer v3)')

# Action constraints (DonkeyCar)
parser.add_argument('--fix_speed', action='store_true', default=True)
parser.add_argument('--throttle_base', type=float, default=0.3)
parser.add_argument('--throttle_min', type=float, default=0.1)
parser.add_argument('--throttle_max', type=float, default=0.5)
parser.add_argument('--angle_min', type=float, default=-1)
parser.add_argument('--angle_max', type=float, default=1)
parser.add_argument('--action_size', default=2)
parser.add_argument('--grayscale', action='store_true', default=False,
                    help='Use 1-channel grayscale input. Default: 3-channel RGB.')
parser.add_argument('--observation_size', default=None)  # set automatically from --grayscale below

# Simulator / connection
parser.add_argument('--sim_path', type=str, default='self')
parser.add_argument('--port', type=int, default=9091)
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--use_visual_reward', action='store_true', default=False,
                    help='Replace CTE telemetry reward with image-based visual CTE proxy'
                         ' (use when sim telemetry is unavailable or for real-world transfer)')
parser.add_argument('--human_override', action='store_true', default=False,
                    help='Human override: operator presses = to stop (off-track) '
                         'or R to reset (clean lap). Removes all dependence on CTE telemetry.')
parser.add_argument('--smooth_weight', type=float, default=0.05,
                    help='Penalty weight for steering jerk: reward -= smooth_weight * |steer_t - steer_{t-1}|. '
                         'No sensor needed. Set 0 to disable.')

# Evaluation & checkpointing
parser.add_argument('--test', action='store_true')
parser.add_argument('--test-interval', type=int, default=5)
parser.add_argument('--test-episodes', type=int, default=1)
parser.add_argument('--checkpoint-interval', type=int, default=500)
parser.add_argument('--checkpoint-experience', action='store_true')
parser.add_argument('--models', type=str, default='')
parser.add_argument('--experience-replay', type=str, default='')
parser.add_argument('--render', action='store_true')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--augment', action='store_true', default=False,
                    help='Apply sim-to-real augmentations during world-model training '
                         '(brightness/contrast, shadow, blur, noise, gamma, erase, crop). '
                         'Recommended for real-world deployment.')

args = parser.parse_args()
args.channels = 1 if args.grayscale else 3
args.observation_size = (args.channels, 64, 64)

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
results_dir = os.path.join('results', args.env, str(args.seed))
images_dir  = os.path.join(results_dir, 'images')
videos_dir  = os.path.join(results_dir, 'videos')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(images_dir,  exist_ok=True)
os.makedirs(videos_dir,  exist_ok=True)

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

metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [],
    'test_episodes': [], 'test_rewards': [],
    'observation_loss': [], 'reward_loss': [], 'kl_loss': [],
    'actor_loss': [], 'value_loss': [],
    'episode_lengths': [], 'mean_cte': [],
}

human_override = None
if args.human_override:
    from dreamer.envs.human_override import HumanOverride
    human_override = HumanOverride()
    print('Human override active — pygame window open.')
    print('  =         : stop (off-track penalty)')
    print('  ↑ Up      : reset (clean lap)')
    print('  ↓ Down    : quit training')

env = Env(args.env, args.symbolic, args.seed, args.max_episode_length,
          args.action_repeat, args.bit_depth, sim_path=args.sim_path,
          host=args.host, port=args.port, use_visual_reward=args.use_visual_reward,
          human_override=human_override, smooth_weight=args.smooth_weight,
          channels=args.channels)
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
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            action[1] = args.throttle_base
            next_observation, reward, done = env.step(action)
            agent.D.append(next_observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(
            t * args.action_repeat + (0 if not metrics['steps'] else metrics['steps'][-1])
        )
        metrics['episodes'].append(s)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for episode in tqdm(
    range(metrics['episodes'][-1] + 1, args.episodes + 1),
    total=args.episodes,
    initial=metrics['episodes'][-1] + 1,
):
    if human_override and human_override.should_quit:
        print('Quit requested via human override — stopping training.')
        break

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
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)

    # --- Data collection ---
    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief = torch.zeros(1, args.belief_size, device=args.device)
        posterior_state = torch.zeros(1, args.state_size, device=args.device)
        action = torch.zeros(1, env.action_size, device=args.device)
        cte_history = []

        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            belief, posterior_state = agent.infer_state(
                observation.to(device=args.device), action, belief, posterior_state
            )
            action = agent.select_action((belief, posterior_state), deterministic=False)
            next_observation, reward, done = env.step(
                action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
            )
            agent.D.append(next_observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if hasattr(env, 'last_cte') and not args.human_override:
                cte_history.append(abs(env.last_cte))
            if args.render:
                env.render()
            if done:
                pbar.close()
                break

    ep_len = t + 1
    cte_arr = np.array(cte_history) if cte_history else np.array([0.0])
    metrics['steps'].append(t + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    metrics['episode_lengths'].append(ep_len)
    metrics['mean_cte'].append(float(cte_arr.mean()))
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)
    lineplot(metrics['episodes'][-len(metrics['mean_cte']):], metrics['mean_cte'], 'mean_cte', results_dir)

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
        agent.transition_model.eval()
        agent.observation_model.eval()
        agent.reward_model.eval()
        agent.encoder.eval()
        agent.actor_model.eval()
        agent.value_model.eval()

        with torch.no_grad():
            observation = env.reset()
            total_rewards = 0
            video_frames = []
            belief = torch.zeros(args.test_episodes, args.belief_size, device=args.device)
            posterior_state = torch.zeros(args.test_episodes, args.state_size, device=args.device)
            action = torch.zeros(args.test_episodes, env.action_size, device=args.device)

            for t in tqdm(range(args.max_episode_length // args.action_repeat)):
                belief, posterior_state = agent.infer_state(
                    observation.to(device=args.device), action, belief, posterior_state
                )
                action = agent.select_action((belief, posterior_state), deterministic=True)
                next_observation, reward, done = env.step(
                    action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
                )
                total_rewards += reward
                if not args.symbolic:
                    video_frames.append(
                        make_grid(
                            torch.cat([observation, agent.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
                            nrow=5,
                        ).numpy()
                    )
                observation = next_observation
                if done:
                    break

        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards)
        lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
        lineplot(
            np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1],
            metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step',
        )
        if not args.symbolic:
            episode_str = str(episode).zfill(len(str(args.episodes)))
            write_video(video_frames, 'ep_%s' % episode_str, videos_dir)
            frame = torch.as_tensor(video_frames[-1])
            save_image(frame, os.path.join(images_dir, 'ep_%s.png' % episode_str))
            save_image(frame, os.path.join(images_dir, 'latest.png'))
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        agent.transition_model.train()
        agent.observation_model.train()
        agent.reward_model.train()
        agent.encoder.train()
        agent.actor_model.train()
        agent.value_model.train()

    print('episodes: {}, total_steps: {}, train_reward: {}'.format(
        metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]
    ))

    # --- Checkpoint ---
    if episode % args.checkpoint_interval == 0:
        torch.save({
            'transition_model': agent.transition_model.state_dict(),
            'observation_model': agent.observation_model.state_dict(),
            'reward_model': agent.reward_model.state_dict(),
            'encoder': agent.encoder.state_dict(),
            'actor_model': agent.actor_model.state_dict(),
            'value_model': agent.value_model.state_dict(),
            'value_model2': agent.value_model2.state_dict(),
            'world_optimizer': agent.world_optimizer.state_dict(),
            'actor_optimizer': agent.actor_optimizer.state_dict(),
            'value_optimizer': agent.value_optimizer.state_dict(),
        }, os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            torch.save(agent.D, os.path.join(results_dir, 'experience.pth'))

env.close()
csv_file.close()
print(f'Rewards saved to {csv_path}')
