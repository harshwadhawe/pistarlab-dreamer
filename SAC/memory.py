import numpy as np
import torch


class ReplayBuffer:
    """
    Circular transition buffer storing (obs, action, reward, done, next_obs).
    obs stored as uint8 (same encoding as Dreamer's comms) to halve memory for RGB.
    """

    def __init__(self, capacity, obs_shape, action_size, device):
        C, H, W = obs_shape
        self.capacity = capacity
        self.device   = device
        self.ptr = self.size = 0

        self.obs      = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.actions  = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=bool)

    @staticmethod
    def _to_uint8(obs):
        return ((obs + 0.5) * 255).clip(0, 255).astype(np.uint8)

    def _to_tensor(self, arr):
        return torch.from_numpy(arr.copy()).float().to(self.device) / 255.0 - 0.5

    def add(self, obs, action, reward, done, next_obs):
        self.obs[self.ptr]      = self._to_uint8(obs)
        self.next_obs[self.ptr] = self._to_uint8(next_obs)
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, obs_seq, actions, rewards, dones):
        T = len(rewards)
        for t in range(T):
            self.add(
                obs_seq[t], actions[t], rewards[t], dones[t],
                obs_seq[min(t + 1, T - 1)],
            )

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            self._to_tensor(self.obs[idx]),
            self._to_tensor(self.next_obs[idx]),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )

    def __len__(self):
        return self.size
