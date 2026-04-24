import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """
    SAC actor: z → TanhNormal → (action, log_prob).

    Steering:  tanh output ∈ (-1, 1)
    Throttle:  tanh output scaled → (throttle_min, throttle_max)
    log_prob includes tanh squashing correction + throttle affine Jacobian.
    """

    LOG_STD_MIN, LOG_STD_MAX = -5, 2

    def __init__(self, z_dim, action_size, hidden_size,
                 fix_speed=False, throttle_base=0.3,
                 throttle_min=0.25, throttle_max=0.35):
        super().__init__()
        self.fix_speed      = fix_speed
        self.throttle_base  = throttle_base
        self.throttle_loc   = (throttle_max + throttle_min) / 2
        self.throttle_scale = (throttle_max - throttle_min) / 2

        out = 2 * (action_size - 1) if fix_speed else 2 * action_size
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, out),
        )

    def forward(self, z, deterministic=False):
        mean, log_std = self.net(z).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

        dist = Normal(mean, log_std.exp())
        u = mean if deterministic else dist.rsample()
        a = torch.tanh(u)

        log_prob = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)

        if self.fix_speed:
            throttle = torch.full((a.size(0), 1), self.throttle_base, device=a.device)
            action = torch.cat([a, throttle], dim=-1)
        else:
            steer    = a[:, :1]
            throttle = a[:, 1:2] * self.throttle_scale + self.throttle_loc
            action   = torch.cat([steer, throttle], dim=-1)
            log_prob = log_prob - torch.log(
                torch.tensor(self.throttle_scale, dtype=torch.float32, device=a.device)
            )

        return action, log_prob


class Critic(nn.Module):
    """Twin Q-networks: Q(z, a) → scalar each. Takes action as input unlike Dreamer's V(b,s)."""

    def __init__(self, z_dim, action_size, hidden_size):
        super().__init__()
        inp = z_dim + action_size

        def _mlp():
            return nn.Sequential(
                nn.Linear(inp, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, z, action):
        q1, q2 = self(z, action)
        return torch.min(q1, q2)
