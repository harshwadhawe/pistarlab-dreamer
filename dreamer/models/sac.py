"""
SAC from pixels — same CNN encoder as Dreamer for fair comparison.

Encoder trained through critic (Bellman) loss.
Actor operates on stop-gradient encoder features.
Twin Q-critics + auto-tuned temperature.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class SACEncoder(nn.Module):
    """4-layer CNN matching Dreamer's VisualEncoder architecture."""
    def __init__(self, channels=1, embedding_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32,  4, stride=2), nn.ReLU(),
            nn.Conv2d(32,       64,  4, stride=2), nn.ReLU(),
            nn.Conv2d(64,       128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128,      256, 4, stride=2), nn.ReLU(),
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, obs):
        x = self.net(obs).view(obs.size(0), -1)
        return self.fc(x)


class SACActor(nn.Module):
    """Tanh-squashed Gaussian actor on top of encoder embedding."""
    def __init__(self, embedding_size, hidden_size, fix_speed,
                 throttle_base, throttle_min, throttle_max, angle_min, angle_max):
        super().__init__()
        self.fix_speed     = fix_speed
        self.throttle_base = throttle_base
        self.throttle_min  = throttle_min
        self.throttle_max  = throttle_max
        self.angle_min     = angle_min
        self.angle_max     = angle_max
        predict_dims = 1 if fix_speed else 2
        self.net      = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size,    hidden_size), nn.ELU(),
        )
        self.mean_head    = nn.Linear(hidden_size, predict_dims)
        self.log_std_head = nn.Linear(hidden_size, predict_dims)

    def forward(self, emb, deterministic=False):
        h       = self.net(emb)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5, 2)
        std     = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        raw  = mean if deterministic else dist.rsample()
        tanh = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1 - tanh.pow(2) + 1e-6)).sum(-1, keepdim=True)

        if self.fix_speed:
            half = (self.angle_max - self.angle_min) / 2
            steer    = tanh * half
            throttle = torch.full_like(steer, self.throttle_base)
        else:
            steer    = tanh[:, :1] * (self.angle_max - self.angle_min) / 2
            t_mid    = (self.throttle_max + self.throttle_min) / 2
            t_range  = (self.throttle_max - self.throttle_min) / 2
            throttle = tanh[:, 1:] * t_range + t_mid

        return torch.cat([steer, throttle], dim=-1), log_prob


class SACCritic(nn.Module):
    """Single Q-network — instantiate twice for twin critics."""
    def __init__(self, embedding_size, hidden_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size + action_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, emb, action):
        return self.net(torch.cat([emb, action], dim=-1))


# ---------------------------------------------------------------------------
# Replay buffer — stores obs as uint8 to save memory
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_size, device):
        self.capacity = capacity
        self.device   = device
        self.ptr = self.size = 0
        # Store as uint8 (0-255) — obs in [-0.5, 0.5] → *255+127 → uint8
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),           dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),           dtype=np.float32)

    @staticmethod
    def _to_uint8(obs):
        return ((obs + 0.5) * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _to_float(obs):
        return obs.astype(np.float32) / 255.0 - 0.5

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = self._to_uint8(obs)
        self.next_obs[self.ptr] = self._to_uint8(next_obs)
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self._to_float(self.obs[idx])).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self._to_float(self.next_obs[idx])).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------

class SACAgent:
    def __init__(self, args, obs_shape, action_size):
        self.args        = args
        self.device      = args.device
        self.action_size = action_size
        channels         = obs_shape[0]
        predict_dims     = 1 if args.fix_speed else 2

        self.encoder    = SACEncoder(channels, args.embedding_size).to(self.device)
        self.actor      = SACActor(args.embedding_size, args.hidden_size,
                                   args.fix_speed, args.throttle_base,
                                   args.throttle_min, args.throttle_max,
                                   args.angle_min, args.angle_max).to(self.device)
        self.critic1     = SACCritic(args.embedding_size, args.hidden_size, action_size).to(self.device)
        self.critic2     = SACCritic(args.embedding_size, args.hidden_size, action_size).to(self.device)
        self.critic1_tgt = SACCritic(args.embedding_size, args.hidden_size, action_size).to(self.device)
        self.critic2_tgt = SACCritic(args.embedding_size, args.hidden_size, action_size).to(self.device)
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        self.target_entropy = float(-predict_dims)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)

        # Encoder + critics trained together via Bellman loss
        critic_params = (list(self.encoder.parameters()) +
                         list(self.critic1.parameters()) +
                         list(self.critic2.parameters()))
        self.critic_opt = torch.optim.Adam(critic_params, lr=args.world_lr, eps=args.adam_epsilon)
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=args.adam_epsilon)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=args.actor_lr, eps=args.adam_epsilon)

        self.D = ReplayBuffer(args.experience_size, obs_shape, action_size, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            emb = self.encoder(obs.to(self.device))
            action, _ = self.actor(emb, deterministic=deterministic)
        return action.cpu()

    def update(self, batch_size, n_steps):
        if len(self.D) < batch_size:
            return 0.0, 0.0, 0.0

        critic_losses, actor_losses = [], []

        for _ in range(n_steps):
            obs, actions, rewards, next_obs, dones = self.D.sample(batch_size)

            # --- Critic update (trains encoder) ---
            with torch.no_grad():
                next_emb = self.encoder(next_obs)
                next_act, next_log_prob = self.actor(next_emb)
                q1_next = self.critic1_tgt(next_emb, next_act)
                q2_next = self.critic2_tgt(next_emb, next_act)
                q_target = rewards + (1 - dones) * self.args.discount * (
                    torch.min(q1_next, q2_next) - self.alpha * next_log_prob
                )

            emb  = self.encoder(obs)
            q1   = self.critic1(emb, actions)
            q2   = self.critic2(emb, actions)
            c_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.critic1.parameters()) + list(self.critic2.parameters()),
                self.args.grad_clip_norm,
            )
            self.critic_opt.step()
            critic_losses.append(c_loss.item())

            # --- Actor update (stop-grad on encoder) ---
            emb_sg = self.encoder(obs).detach()
            act_new, log_prob = self.actor(emb_sg)
            q1_new = self.critic1(emb_sg, act_new)
            q2_new = self.critic2(emb_sg, act_new)
            a_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm)
            self.actor_opt.step()
            actor_losses.append(a_loss.item())

            # --- Temperature update ---
            al_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            al_loss.backward()
            self.alpha_opt.step()

            # --- Soft target update ---
            for p, pt in zip(self.critic1.parameters(), self.critic1_tgt.parameters()):
                pt.data.mul_(1 - self.args.polyak).add_(self.args.polyak * p.data)
            for p, pt in zip(self.critic2.parameters(), self.critic2_tgt.parameters()):
                pt.data.mul_(1 - self.args.polyak).add_(self.args.polyak * p.data)

        return np.mean(critic_losses), np.mean(actor_losses), self.alpha

    def save_checkpoint(self, path):
        torch.save({
            'encoder':   self.encoder.state_dict(),
            'actor':     self.actor.state_dict(),
            'critic1':   self.critic1.state_dict(),
            'critic2':   self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.actor.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.log_alpha = ckpt['log_alpha']
