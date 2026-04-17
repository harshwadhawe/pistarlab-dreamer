import collections
import cv2
import numpy as np
import torch

from ..utils.obs import preprocess_frame


GYM_ENVS = [
    'Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2',
    'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2',
]
CONTROL_SUITE_ENVS = [
    'cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin',
    'cheetah-run', 'ball_in_cup-catch', 'walker-walk', 'reacher-hard',
    'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup',
]
CONTROL_SUITE_ACTION_REPEATS = {
    'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4,
    'ball_in_cup': 6, 'walker': 2, 'humanoid': 2, 'fish': 2, 'acrobot': 4,
}
DONKEY_CAR_ENVS = [
    'donkey-warehouse-v0', 'donkey-generated-roads-v0', 'donkey-avc-sparkfun-v0',
    'donkey-generated-track-v0', 'donkey-mountain-track-v0',
    'donkey-roboracingleague-track-v0', 'donkey-minimonaco-track-v0',
    'donkey-thunderhill-track-v0', 'donkey-warren-track-v0',
    'donkey-circuit-launch-track-v0', 'donkey-waveshare-v0',
]


def preprocess_observation_(observation, bit_depth):
    """Quantise to given bit depth and centre inplace (float32 [0,255] → [-0.5, 0.5])."""
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


def postprocess_observation(observation, bit_depth):
    """Postprocess for storage (float32 [-0.5, 0.5] → uint8 [0, 255])."""
    return np.clip(
        np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth),
        0, 2 ** 8 - 1,
    ).astype(np.uint8)


def _images_to_observation(images, bit_depth, channels=1):
    """Preprocess raw camera frame → batched torch tensor (1, C, 64, 64) in [-0.5, 0.5]."""
    obs = preprocess_frame(images, channels)          # (C, 64, 64) numpy
    return torch.as_tensor(obs).unsqueeze(0)          # (1, C, 64, 64) torch


class ControlSuiteEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self.symbolic = symbolic
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        if not symbolic:
            self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print('Using action repeat %d; recommended for domain is %d' % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(
                np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs
                                for obs in state.observation.values()], axis=0),
                dtype=torch.float32,
            ).unsqueeze(dim=0)
        return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for _ in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1
            done = state.last() or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(
                np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs
                                for obs in state.observation.values()], axis=0),
                dtype=torch.float32,
            ).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
        return observation, reward, done

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0])
                    for obs in self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class GymEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        import gym
        self.symbolic = symbolic
        self._env = gym.make(env)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
        return observation, reward, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


def _visual_cte(frame_bgr):
    """
    Estimate lateral track offset from the camera image.
    Used as a CTE proxy when --use_visual_reward is set (no sim telemetry needed).

    Uses Canny edge detection to find the left/right track boundaries and
    returns the normalised offset of their midpoint from the image centre.

    Returns float in [-1, 1]: 0 = centred, ±1 = at track edge.
    Returns None if track boundaries cannot be detected.
    """
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * 0.4):, :]   # bottom 60% — avoids horizon clutter
    roi_w = roi.shape[1]

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    col_sum = edges.sum(axis=0).astype(np.float32)
    if col_sum.max() == 0:
        return None

    mid = roi_w // 2
    left_col  = int(np.argmax(col_sum[:mid]))
    right_col = mid + int(np.argmax(col_sum[mid:]))

    if col_sum[left_col] < 100 or col_sum[right_col] < 100:
        return None   # edges too weak to be reliable

    lane_cx = (left_col + right_col) / 2.0
    return float((lane_cx - mid) / mid)


class DonkeyCarEnv:
    # Stuck detection: if speed stays below this threshold for this many consecutive
    # steps, force episode termination (car parked against a wall within CTE limit).
    STUCK_SPEED_THRESHOLD = 0.1
    STUCK_STEPS_LIMIT = 15   # consecutive low-speed steps before forced done

    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth,
                 sim_path, host='127.0.0.1', port=9091, use_visual_reward=False,
                 controller=None, smooth_weight=0.0, smooth_window=10, channels=1):
        import gymnasium as gym
        import gym_donkeycar  # registers envs with gymnasium
        self.symbolic = symbolic
        self._seed = seed
        self._first_reset = True
        self.use_visual_reward = use_visual_reward
        self.controller = controller
        self.discard_requested = False
        conf = {'host': host, 'port': port, 'max_cte': 4}
        if sim_path != 'self':
            conf['exe_path'] = sim_path
        self._env = gym.make(env, conf=conf)

        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.last_cte = 0.0
        self.last_speed = 0.0
        self.last_hit = False
        self._stuck_steps = 0
        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._episode_num    = 0
        self.smooth_weight   = smooth_weight
        self.smooth_window   = smooth_window
        self._steer_buf      = collections.deque(maxlen=smooth_window)
        self._channels       = channels

    def reset(self):
        self.t = 0
        self.last_cte = 0.0
        self._stuck_steps = 0
        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._episode_num   += 1
        self._steer_buf.clear()
        self.discard_requested = False
        # Pass seed only on the very first reset so subsequent episodes
        # use the env's internal seeded RNG rather than resetting to the same state.
        if self._first_reset:
            obs, _ = self._env.reset(seed=self._seed)
            self._first_reset = False
        else:
            obs, _ = self._env.reset()
        return _images_to_observation(obs, self.bit_depth, channels=self._channels)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, terminated, truncated, info = self._env.step(action)

            speed = float(info.get('speed', 0.0))
            hit   = info.get('hit', 'none') != 'none'

            if self.controller is not None:
                # Human operator decides BOTH reward and termination.
                # Suppress sim's CTE-based terminated/reward entirely.
                ev = self.controller.consume_event()
                reward_k, terminated, discard = self.controller.event_to_outcome(ev)
                if discard:
                    self.discard_requested = True
            elif self.use_visual_reward:
                visual_cte = _visual_cte(state)
                if visual_cte is None:
                    reward_k = -1.0
                    terminated = True
                else:
                    reward_k = (1.0 - abs(visual_cte)) * max(speed, 0.0)
                    self.last_cte = visual_cte * 4.0
                    terminated = terminated or abs(visual_cte) > 0.9
            else:
                self.last_cte = float(info.get('cte', 0.0))

            if self.controller is None:
                # Automatic termination guards (not needed when human is watching)
                if hit:
                    terminated = True
                if speed < self.STUCK_SPEED_THRESHOLD:
                    self._stuck_steps += 1
                else:
                    self._stuck_steps = 0
                if self._stuck_steps >= self.STUCK_STEPS_LIMIT:
                    terminated = True

            # Steering jitter penalty: std dev over rolling window penalises
            # sustained oscillation more than a single abrupt step change.
            self._steer_buf.append(float(action[0]))
            if self.smooth_weight > 0.0 and not terminated and len(self._steer_buf) >= 2:
                reward_k -= self.smooth_weight * float(np.std(self._steer_buf))

            reward += reward_k
            self.t += 1
            self._episode_steps += 1
            self._episode_reward += reward_k
            self.last_speed = speed
            self.last_hit = hit
            done = terminated or truncated
            if done:
                break
        observation = _images_to_observation(state, self.bit_depth, channels=self._channels)
        return observation, reward, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (self._channels, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    def brake(self):
        """Send zero-throttle to park the car during world-model training."""
        import numpy as np
        stop = np.array([0.0, 0.0], dtype=np.float32)
        try:
            self._env.step(stop)
        except Exception:
            pass  # best-effort — don't crash training if sim drops the packet

    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, sim_path, host, port,
        use_visual_reward=False, controller=None, smooth_weight=0.0, smooth_window=10, channels=1):
    if env in GYM_ENVS:
        return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    elif env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    elif env in DONKEY_CAR_ENVS:
        return DonkeyCarEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth,
                            sim_path, host, port, use_visual_reward=use_visual_reward,
                            controller=controller, smooth_weight=smooth_weight,
                            smooth_window=smooth_window, channels=channels)
    else:
        raise NotImplementedError(f'Unknown environment: {env}')


class EnvBatcher:
    """Wraps multiple environments for parallel rollouts."""

    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    def step(self, actions):
        done_mask = torch.nonzero(torch.tensor(self.dones), as_tuple=False)[:, 0]
        observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
        self.dones = dones
        observations = torch.cat(observations)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones

    def close(self):
        for env in self.envs:
            env.close()
