from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions


class TransitionModel(nn.Module):
    """RSSM core: deterministic GRU belief + stochastic latent state."""

    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size,
                 embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.rnn_norm = nn.LayerNorm(belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Input shapes  (T = sequence length, B = batch):
          prev_state   [B, state_size]
          actions      [T, B, action_size]
          prev_belief  [B, belief_size]
          observations [T, B, embedding_size]   (optional — posterior)
          nonterminals [T, B]                   (optional — mask terminals)

        Returns:
          beliefs, prior_states, prior_means, prior_std_devs
          [+ posterior_states, posterior_means, posterior_std_devs if observations given]
          Each has shape [T, B, *_size].
        """
        T = actions.size(0) + 1
        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_std_devs = [torch.empty(0)] * T

        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state

        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]
            if nonterminals is not None and t != 0:
                _state = _state * nonterminals[t - 1].unsqueeze(dim=-1)

            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn_norm(self.rnn(hidden, beliefs[t]))

            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

            if observations is not None:
                t_ = t - 1
                hidden = self.act_fn(self.fc_embed_belief_posterior(
                    torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)
                ))
                posterior_means[t + 1], _post_std = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_post_std) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])

        result = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            result += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return result



class VisualObservationModel(jit.ScriptModule):
    """
    4-layer transposed CNN decoder: (belief, state) → channels×64×64 image.

    Spatial trace: 1×1 → 5×5 → 13×13 → 30×30 → 64×64

    channels=1  grayscale (phase B, current)
    channels=3  RGB       (phase A — update --observation_size to (3,64,64))
    """

    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size,
                 activation_function='relu', channels=1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1   = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64,  5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64,  32,  6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32,  channels, 6, stride=2)

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        return self.conv4(hidden)


def ObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    channels = observation_size[0]
    return VisualObservationModel(belief_size, state_size, embedding_size, activation_function, channels=channels)



class VisualEncoder(jit.ScriptModule):
    """
    4-layer CNN: channels×64×64 → 1024-D embedding.

    Spatial trace: 64×64 → 31×31 → 14×14 → 6×6 → 2×2 → flat 1024

    channels=1  grayscale (phase B, current)
    channels=3  RGB       (phase A — update --observation_size to (3,64,64))
    """

    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, activation_function='relu', channels=1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(channels, 32,  4, stride=2)
        self.conv2 = nn.Conv2d(32,        64,  4, stride=2)
        self.conv3 = nn.Conv2d(64,       128,  4, stride=2)
        self.conv4 = nn.Conv2d(128,      256,  4, stride=2)
        self.fc    = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        return self.fc(hidden)


def Encoder(observation_size, embedding_size, activation_function='relu'):
    channels = observation_size[0]
    return VisualEncoder(embedding_size, activation_function, channels=channels)


class RewardModel(jit.ScriptModule):
    """MLP: (belief, state) → scalar reward."""

    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        return self.fc3(hidden).squeeze(dim=-1)


class PCONTModel(nn.Module):
    """Predicts continuation probability (Bernoulli) for terminal state handling."""

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
        return torch.sigmoid(self.fc4(hidden).squeeze(dim=1))
