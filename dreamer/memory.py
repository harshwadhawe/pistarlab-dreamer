import numpy as np
import torch


class ExperienceReplay:
    """Circular replay buffer storing (obs, action, reward, nonterminal) tuples.

    Observations stored as float32 in [-0.5, 0.5].
    sample() returns contiguous chunks of shape [chunk, batch, ...].
    """

    def __init__(self, size, observation_size, action_size, device):
        self.device = device
        self.size = size
        self.observations = np.empty((size, *observation_size), dtype=np.float32)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.nonterminals = np.empty((size,), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

    def snapshot(self):
        """Return a lightweight checkpoint of buffer pointers (not data)."""
        return (self.idx, self.steps, self.episodes, self.full)

    def restore(self, snap):
        """Roll back to a previous snapshot, discarding appended entries."""
        self.idx, self.steps, self.episodes, _ = snap
        self.full = self.steps >= self.size

    def append(self, observation, action, reward, done):
        self.observations[self.idx] = observation.numpy()
        self.actions[self.idx] = action.numpy() if isinstance(action, torch.Tensor) else action
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes += 1 if done else 0

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = self.idx not in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        return (
            observations.reshape(L, n, *observations.shape[1:]),
            self.actions[vec_idxs].reshape(L, n, -1),
            self.rewards[vec_idxs].reshape(L, n),
            self.nonterminals[vec_idxs].reshape(L, n),
        )

    def sample(self, n, L):
        available = self.size if self.full else self.idx
        if available < L:
            raise RuntimeError(
                f"Replay buffer has {available} steps but chunk_size={L}. "
                f"Collect more seed episodes before training."
            )
        batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]
