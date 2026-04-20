import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution


class SampleDist:
    """
    Approximates mean/mode/entropy of a TransformedDistribution via Monte-Carlo
    sampling, since those properties become invalid after transformation.
    """

    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        return torch.mean(dist.rsample(), 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return -torch.mean(dist.log_prob(sample), 0)


class ActorModel(nn.Module):
    """
    4-layer MLP policy.
    Outputs a TanhTransformed Normal distribution over actions.
    When fix_speed=True, only steering is predicted; throttle is set to throttle_base.
    """

    def __init__(self, action_size, belief_size, state_size, hidden_size,
                 mean_scale=5, min_std=1e-4, init_std=5,
                 activation_function='elu', fix_speed=False, throttle_base=0.3,
                 throttle_min=0.1, throttle_max=0.5):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fix_speed = fix_speed
        self.throttle_base = throttle_base
        self.throttle_loc   = (throttle_max + throttle_min) / 2
        self.throttle_scale = (throttle_max - throttle_min) / 2
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale

        out_size = 2 * (action_size - 1) if fix_speed else 2 * action_size
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, out_size)

    def forward(self, belief, state, deterministic=False, with_logprob=False):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=-1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        hidden = self.fc5(hidden)

        mean, std = torch.chunk(hidden, 2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + raw_init_std) + self.min_std

        dist = torch.distributions.Normal(mean, std)

        if self.fix_speed:
            transform = [AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)]
        else:
            device = mean.device
            transform = [
                AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.),
                AffineTransform(
                    loc=torch.tensor([0.0, self.throttle_loc]).to(device),
                    scale=torch.tensor([1.0, self.throttle_scale]).to(device),
                ),
            ]

        dist = SampleDist(TransformedDistribution(dist, transform))

        action = dist.mean if deterministic else dist.rsample()
        logp_pi = dist.log_prob(action).sum(dim=1) if with_logprob else None

        if self.fix_speed:
            throttle = torch.full((action.shape[0], 1), self.throttle_base,
                                  dtype=action.dtype, device=action.device)
            action = torch.cat((action, throttle), dim=-1)

        return action, logp_pi


class ValueModel(nn.Module):
    """4-layer MLP critic: (belief, state) → scalar value."""

    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        return self.fc4(hidden).squeeze(dim=1)
