import collections
import numpy as np
import torch

from ..utils.obs import preprocess_frame


DONKEY_CAR_ENVS = [
    'donkey-warehouse-v0', 'donkey-generated-roads-v0', 'donkey-avc-sparkfun-v0',
    'donkey-generated-track-v0', 'donkey-mountain-track-v0',
    'donkey-roboracingleague-track-v0', 'donkey-minimonaco-track-v0',
    'donkey-thunderhill-track-v0', 'donkey-warren-track-v0',
    'donkey-circuit-launch-track-v0', 'donkey-waveshare-v0',
]


def _images_to_observation(images, channels=1):
    """Preprocess raw camera frame → batched torch tensor (1, C, 64, 64) in [-0.5, 0.5]."""
    obs = preprocess_frame(images, channels)
    return torch.as_tensor(obs).unsqueeze(0)


class DonkeyCarEnv:
    STUCK_SPEED_THRESHOLD = 0.1
    STUCK_STEPS_LIMIT = 15

    def __init__(self, env, seed, max_episode_length,
                 sim_path, host='127.0.0.1', port=9091,
                 controller=None, smooth_weight=0.0, smooth_window=10, channels=1):
        import gymnasium as gym
        import gym_donkeycar  # registers envs with gymnasium
        self._seed = seed
        self._first_reset = True
        self.controller = controller
        self.discard_requested = False
        conf = {'host': host, 'port': port, 'max_cte': 4}
        if sim_path != 'self':
            conf['exe_path'] = sim_path
        self._env = gym.make(env, conf=conf)

        self.max_episode_length = max_episode_length
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
        if self._first_reset:
            obs, _ = self._env.reset(seed=self._seed)
            self._first_reset = False
        else:
            obs, _ = self._env.reset()
        return _images_to_observation(obs, channels=self._channels)

    def step(self, action):
        action = action.detach().numpy()
        state, reward_k, terminated, truncated, info = self._env.step(action)

        speed = float(info.get('speed', 0.0))
        hit   = info.get('hit', 'none') != 'none'

        if self.controller is not None:
            ev = self.controller.consume_event()
            reward_k, terminated, discard = self.controller.event_to_outcome(ev)
            if discard:
                self.discard_requested = True
        else:
            self.last_cte = float(info.get('cte', 0.0))
            if hit:
                terminated = True
            if speed < self.STUCK_SPEED_THRESHOLD:
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0
            if self._stuck_steps >= self.STUCK_STEPS_LIMIT:
                terminated = True

        self._steer_buf.append(float(action[0]))
        if self.smooth_weight > 0.0 and not terminated and len(self._steer_buf) >= 2:
            reward_k -= self.smooth_weight * float(np.std(self._steer_buf))

        self.t += 1
        self._episode_steps += 1
        self._episode_reward += reward_k
        self.last_speed = speed
        self.last_hit = hit
        done = terminated or truncated
        observation = _images_to_observation(state, channels=self._channels)
        return observation, reward_k, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return (self._channels, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    def brake(self):
        """Send zero-throttle to park the car during world-model training."""
        stop = np.array([0.0, 0.0], dtype=np.float32)
        try:
            self._env.step(stop)
        except Exception:
            pass

    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


def Env(env, seed, max_episode_length, sim_path, host, port,
        controller=None, smooth_weight=0.0, smooth_window=10, channels=1):
    if env in DONKEY_CAR_ENVS:
        return DonkeyCarEnv(env, seed, max_episode_length,
                            sim_path, host, port,
                            controller=controller, smooth_weight=smooth_weight,
                            smooth_window=smooth_window, channels=channels)
    raise NotImplementedError(f'Unknown environment: {env}')
