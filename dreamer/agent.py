from copy import deepcopy

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions.independent import Independent
from torch.nn import functional as F
from tqdm import tqdm

from .memory import ExperienceReplay
from .models import (
    bottle, Encoder, ObservationModel, RewardModel,
    TransitionModel, ValueModel, ActorModel, PCONTModel,
)
from .utils.math_utils import cal_returns


def count_vars(module):
    return sum(np.prod(p.shape) for p in module.parameters())


class Dreamer:
    """
    Dreamer agent: world model (RSSM) + SAC-style actor-critic trained in latent imagination.
    All models, optimizers, and replay buffer are owned here.
    """

    def __init__(self, args):
        self.args = args

        self.transition_model = TransitionModel(
            args.belief_size, args.state_size, args.action_size,
            args.hidden_size, args.embedding_size, args.dense_act,
        ).to(device=args.device)

        self.observation_model = ObservationModel(
            args.symbolic, args.observation_size,
            args.belief_size, args.state_size, args.embedding_size,
            activation_function=(args.dense_act if args.symbolic else args.cnn_act),
        ).to(device=args.device)

        self.reward_model = RewardModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act,
        ).to(device=args.device)

        self.encoder = Encoder(
            args.symbolic, args.observation_size, args.embedding_size, args.cnn_act,
        ).to(device=args.device)

        self.actor_model = ActorModel(
            args.action_size, args.belief_size, args.state_size, args.hidden_size,
            activation_function=args.dense_act,
            fix_speed=args.fix_speed,
            throttle_base=args.throttle_base,
        ).to(device=args.device)

        self.value_model = ValueModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act,
        ).to(device=args.device)

        self.value_model2 = ValueModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act,
        ).to(device=args.device)

        self.pcont_model = PCONTModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act,
        ).to(device=args.device)

        self.target_value_model = deepcopy(self.value_model)
        self.target_value_model2 = deepcopy(self.value_model2)
        for p in self.target_value_model.parameters():
            p.requires_grad = False
        for p in self.target_value_model2.parameters():
            p.requires_grad = False

        self.world_param = (
            list(self.transition_model.parameters())
            + list(self.observation_model.parameters())
            + list(self.reward_model.parameters())
            + list(self.encoder.parameters())
        )
        if args.pcont:
            self.world_param += list(self.pcont_model.parameters())

        self.world_optimizer = optim.Adam(self.world_param, lr=args.world_lr)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=args.actor_lr)
        self.value_optimizer = optim.Adam(
            list(self.value_model.parameters()) + list(self.value_model2.parameters()),
            lr=args.value_lr,
        )

        self.free_nats = torch.full((1,), args.free_nats, dtype=torch.float32, device=args.device)

        self.D = ExperienceReplay(
            args.experience_size, args.symbolic, args.observation_size,
            args.action_size, args.bit_depth, args.device,
        )

        if self.args.auto_temp:
            self.log_temp = torch.zeros(1, requires_grad=True, device=args.device)
            self.target_entropy = -np.prod(
                args.action_size if not args.fix_speed else self.args.action_size - 1
            ).item()
            self.temp_optimizer = optim.Adam([self.log_temp], lr=args.value_lr)

    def process_im(self, images, image_size=None, rgb=None):
        images = cv2.resize(images, (40, 40))
        images = np.dot(images, [0.299, 0.587, 0.114])
        obs = torch.tensor(images, dtype=torch.float32).div_(255.).sub_(0.5).unsqueeze(dim=0)
        return obs.unsqueeze(dim=0)

    def append_buffer(self, new_traj):
        for observation, action, reward, done in new_traj:
            self.D.append(observation, action.cpu(), reward, done)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss_world(self, state, data):
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = state
        observations, rewards, nonterminals = data

        observation_loss = F.mse_loss(
            bottle(self.observation_model, (beliefs, posterior_states)),
            observations,
            reduction='none',
        ).sum(dim=2 if self.args.symbolic else (2, 3, 4)).mean(dim=(0, 1))

        reward_loss = F.mse_loss(
            bottle(self.reward_model, (beliefs, posterior_states)),
            rewards,
            reduction='none',
        ).mean(dim=(0, 1))

        kl_loss = torch.max(
            kl_divergence(
                Independent(Normal(posterior_means, posterior_std_devs), 1),
                Independent(Normal(prior_means, prior_std_devs), 1),
            ),
            self.free_nats,
        ).mean(dim=(0, 1))

        pcont_loss = 0
        if self.args.pcont:
            pcont_loss = F.binary_cross_entropy(
                bottle(self.pcont_model, (beliefs, posterior_states)), nonterminals
            )

        return (
            observation_loss,
            self.args.reward_scale * reward_loss,
            kl_loss,
            self.args.pcont_scale * pcont_loss if self.args.pcont else 0,
        )

    def _compute_loss_actor(self, imag_beliefs, imag_states, imag_ac_logps=None):
        imag_rewards = bottle(self.reward_model, (imag_beliefs, imag_states))
        imag_values = torch.min(
            bottle(self.value_model, (imag_beliefs, imag_states)),
            bottle(self.value_model2, (imag_beliefs, imag_states)),
        )

        with torch.no_grad():
            pcont = (
                bottle(self.pcont_model, (imag_beliefs, imag_states))
                if self.args.pcont
                else self.args.discount * torch.ones_like(imag_rewards)
            )
        pcont = pcont.detach()

        if imag_ac_logps is not None:
            imag_values[1:] -= self.args.temp * imag_ac_logps

        returns = cal_returns(imag_rewards[:-1], imag_values[:-1], imag_values[-1], pcont[:-1], lambda_=self.args.disclam)
        discount = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0).detach()

        assert list(discount.size()) == list(returns.size())
        return -torch.mean(discount * returns)

    def _compute_loss_critic(self, imag_beliefs, imag_states, imag_ac_logps=None):
        with torch.no_grad():
            target_imag_values = torch.min(
                bottle(self.target_value_model, (imag_beliefs, imag_states)),
                bottle(self.target_value_model2, (imag_beliefs, imag_states)),
            )
            imag_rewards = bottle(self.reward_model, (imag_beliefs, imag_states))
            pcont = (
                bottle(self.pcont_model, (imag_beliefs, imag_states))
                if self.args.pcont
                else self.args.discount * torch.ones_like(imag_rewards)
            )
            if imag_ac_logps is not None:
                target_imag_values[1:] -= self.args.temp * imag_ac_logps

        returns = cal_returns(imag_rewards[:-1], target_imag_values[:-1], target_imag_values[-1], pcont[:-1], lambda_=self.args.disclam)
        target_return = returns.detach()

        value_pred = bottle(self.value_model, (imag_beliefs, imag_states))[:-1]
        value_pred2 = bottle(self.value_model2, (imag_beliefs, imag_states))[:-1]

        value_loss = F.mse_loss(value_pred, target_return, reduction='none').mean(dim=(0, 1))
        value_loss += F.mse_loss(value_pred2, target_return, reduction='none').mean(dim=(0, 1))
        return value_loss

    # ------------------------------------------------------------------
    # Latent imagination rollout
    # ------------------------------------------------------------------

    def _latent_imagination(self, beliefs, posterior_states, with_logprob=False):
        chunk_size, batch_size, _ = posterior_states.size()
        flatten_size = chunk_size * batch_size

        posterior_states = posterior_states.detach().reshape(flatten_size, -1)
        beliefs = beliefs.detach().reshape(flatten_size, -1)

        imag_beliefs = [beliefs]
        imag_states = [posterior_states]
        imag_ac_logps = []

        for _ in range(self.args.planning_horizon):
            imag_action, imag_ac_logp = self.actor_model(
                imag_beliefs[-1].detach(),
                imag_states[-1].detach(),
                deterministic=False,
                with_logprob=with_logprob,
            )
            imag_action = imag_action.unsqueeze(dim=0)
            imag_belief, imag_state, _, _ = self.transition_model(
                imag_states[-1], imag_action, imag_beliefs[-1]
            )
            imag_beliefs.append(imag_belief.squeeze(dim=0))
            imag_states.append(imag_state.squeeze(dim=0))
            if with_logprob:
                imag_ac_logps.append(imag_ac_logp.squeeze(dim=0))

        imag_beliefs = torch.stack(imag_beliefs, dim=0).to(self.args.device)
        imag_states = torch.stack(imag_states, dim=0).to(self.args.device)
        if with_logprob:
            imag_ac_logps = torch.stack(imag_ac_logps, dim=0).to(self.args.device)

        return imag_beliefs, imag_states, imag_ac_logps if with_logprob else None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def update_parameters(self, gradient_steps):
        loss_info = []
        for _ in tqdm(range(gradient_steps)):
            observations, actions, rewards, nonterminals = self.D.sample(self.args.batch_size, self.args.chunk_size)

            init_belief = torch.zeros(self.args.batch_size, self.args.belief_size, device=self.args.device)
            init_state = torch.zeros(self.args.batch_size, self.args.state_size, device=self.args.device)

            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(
                init_state, actions, init_belief,
                bottle(self.encoder, (observations,)), nonterminals,
            )

            # --- World model update ---
            world_model_loss = self._compute_loss_world(
                state=(beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs),
                data=(observations, rewards, nonterminals),
            )
            observation_loss, reward_loss, kl_loss, pcont_loss = world_model_loss
            self.world_optimizer.zero_grad()
            (observation_loss + reward_loss + kl_loss + pcont_loss).backward()
            nn.utils.clip_grad_norm_(self.world_param, self.args.grad_clip_norm, norm_type=2)
            self.world_optimizer.step()

            # Freeze world + value params during actor update
            for p in self.world_param:
                p.requires_grad = False
            for p in self.value_model.parameters():
                p.requires_grad = False
            for p in self.value_model2.parameters():
                p.requires_grad = False

            # --- Latent imagination ---
            imag_beliefs, imag_states, imag_ac_logps = self._latent_imagination(
                beliefs, posterior_states, with_logprob=self.args.with_logprob
            )

            # --- Temperature update (auto_temp) ---
            if self.args.auto_temp:
                temp_loss = -(self.log_temp * (imag_ac_logps[0] + self.target_entropy).detach()).mean()
                self.temp_optimizer.zero_grad()
                temp_loss.backward()
                self.temp_optimizer.step()
                self.args.temp = self.log_temp.exp()

            # --- Actor update ---
            actor_loss = self._compute_loss_actor(imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.actor_optimizer.step()

            # Unfreeze
            for p in self.world_param:
                p.requires_grad = True
            for p in self.value_model.parameters():
                p.requires_grad = True
            for p in self.value_model2.parameters():
                p.requires_grad = True

            # --- Critic update ---
            imag_beliefs = imag_beliefs.detach()
            imag_states = imag_states.detach()
            critic_loss = self._compute_loss_critic(imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)
            self.value_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            nn.utils.clip_grad_norm_(self.value_model2.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.value_optimizer.step()

            loss_info.append([
                observation_loss.item(), reward_loss.item(), kl_loss.item(),
                pcont_loss.item() if self.args.pcont else 0,
                actor_loss.item(), critic_loss.item(),
            ])

        # Hard update target value networks
        with torch.no_grad():
            self.target_value_model.load_state_dict(self.value_model.state_dict())
            self.target_value_model2.load_state_dict(self.value_model2.state_dict())

        return loss_info

    # ------------------------------------------------------------------
    # Inference / action selection
    # ------------------------------------------------------------------

    def infer_state(self, observation, action, belief=None, state=None):
        """One-step RSSM posterior update. Returns (belief, posterior_state)."""
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            state,
            action.unsqueeze(dim=0),
            belief,
            self.encoder(observation).unsqueeze(dim=0),
        )
        return belief.squeeze(dim=0), posterior_state.squeeze(dim=0)

    def select_action(self, state, deterministic=False):
        belief, posterior_state = state
        action, _ = self.actor_model(belief, posterior_state, deterministic=deterministic, with_logprob=False)
        if not deterministic and not self.args.with_logprob:
            action = Normal(action, self.args.expl_amount).rsample()
            action[:, 0].clamp_(min=self.args.angle_min, max=self.args.angle_max)
            if self.args.fix_speed:
                action[:, 1] = self.args.throttle_base
            else:
                action[:, 1].clamp_(min=self.args.throttle_min, max=self.args.throttle_max)
        return action

    # ------------------------------------------------------------------
    # Distributed rollout helpers
    # ------------------------------------------------------------------

    def import_parameters(self, params):
        self.encoder.load_state_dict(params['encoder'])
        self.actor_model.load_state_dict(params['policy'])
        self.transition_model.load_state_dict(params['transition'])

    def export_parameters(self):
        params = {
            'encoder': self.encoder.cpu().state_dict(),
            'policy': self.actor_model.cpu().state_dict(),
            'transition': self.transition_model.cpu().state_dict(),
        }
        self.encoder.to(self.args.device)
        self.actor_model.to(self.args.device)
        self.transition_model.to(self.args.device)
        return params
