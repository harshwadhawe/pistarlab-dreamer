import os
from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions.independent import Independent
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from .augmentations import Augmenter
from .memory import ExperienceReplay
from .models import (
    bottle, Encoder, ObservationModel, RewardModel,
    TransitionModel, ValueModel, ActorModel, PCONTModel,
)
from .utils.math_utils import cal_returns, symlog, symexp


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
            args.observation_size,
            args.belief_size, args.state_size, args.embedding_size, args.cnn_act,
        ).to(device=args.device)

        self.reward_model = RewardModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act,
        ).to(device=args.device)

        self.encoder = Encoder(
            args.observation_size, args.embedding_size, args.cnn_act,
        ).to(device=args.device)

        self.actor_model = ActorModel(
            args.action_size, args.belief_size, args.state_size, args.hidden_size,
            activation_function=args.dense_act,
            fix_speed=args.fix_speed,
            throttle_base=args.throttle_base,
            throttle_min=args.throttle_min,
            throttle_max=args.throttle_max,
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
            args.experience_size, args.observation_size,
            args.action_size, args.device,
        )

        self.augmenter = Augmenter(device=args.device) if args.augment else None

        world_params  = sum(np.prod(p.shape) for p in self.world_param)
        actor_params  = sum(np.prod(p.shape) for p in self.actor_model.parameters())
        value_params  = sum(np.prod(p.shape) for p in self.value_model.parameters())
        print(f'[Agent] Model created — world {world_params/1e6:.2f}M  actor {actor_params/1e6:.2f}M  value {value_params/1e6:.2f}M  device={args.device}')

        # Return normalisation EMAs (Dreamer v3)
        self._ret_ema_low = 1.0
        self._ret_ema_high = 1.0
        self._norm_step = 0

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
        ).sum(dim=(2, 3, 4)).mean(dim=(0, 1))

        reward_target = symlog(rewards) if self.args.symlog_rewards else rewards
        reward_loss = F.mse_loss(
            bottle(self.reward_model, (beliefs, posterior_states)),
            reward_target,
            reduction='none',
        ).mean(dim=(0, 1))

        # KL balancing (Dreamer v2): separate dynamics loss and representation loss
        if self.args.kl_balance:
            kl_lhs = kl_divergence(
                Independent(Normal(posterior_means.detach(), posterior_std_devs.detach()), 1),
                Independent(Normal(prior_means, prior_std_devs), 1),
            )  # trains prior / dynamics model
            kl_rhs = kl_divergence(
                Independent(Normal(posterior_means, posterior_std_devs), 1),
                Independent(Normal(prior_means.detach(), prior_std_devs.detach()), 1),
            )  # trains posterior / encoder
            kl_loss = (
                0.8 * torch.max(kl_lhs, self.free_nats) +
                0.2 * torch.max(kl_rhs, self.free_nats)
            ).mean(dim=(0, 1))
        else:
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
        if self.args.symlog_rewards:
            imag_rewards = symexp(imag_rewards)

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

        # Return normalisation (Dreamer v3): recompute percentiles every 10 steps.
        # EMA smooths the scale so stale percentiles are fine between updates.
        if self.args.return_norm:
            self._norm_step += 1
            if self._norm_step % 10 == 1:
                with torch.no_grad():
                    flat = returns.flatten()
                    n = flat.numel()
                    p5  = torch.kthvalue(flat, max(1, int(0.05 * n))).values.item()
                    p95 = torch.kthvalue(flat, max(1, int(0.95 * n))).values.item()
                self._ret_ema_low  = 0.99 * self._ret_ema_low  + 0.01 * p5
                self._ret_ema_high = 0.99 * self._ret_ema_high + 0.01 * p95
            S = max(1.0, self._ret_ema_high - self._ret_ema_low)
            returns = returns / S

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
            if self.args.symlog_rewards:
                imag_rewards = symexp(imag_rewards)
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

    def _opt_step(self, loss, optimizer, params_for_clip):
        loss.backward()
        nn.utils.clip_grad_norm_(params_for_clip, self.args.grad_clip_norm, norm_type=2)
        optimizer.step()

    def update_parameters(self, gradient_steps):
        loss_info = []
        for _ in tqdm(range(gradient_steps)):
            observations, actions, rewards, nonterminals = self.D.sample(self.args.batch_size, self.args.chunk_size)

            if self.augmenter is not None:
                observations = self.augmenter(observations)

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
            self._opt_step(observation_loss + reward_loss + kl_loss + pcont_loss,
                           self.world_optimizer, self.world_param)

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

            # --- Actor update ---
            actor_loss = self._compute_loss_actor(imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)
            self.actor_optimizer.zero_grad()
            self._opt_step(actor_loss, self.actor_optimizer, self.actor_model.parameters())

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
            self._opt_step(critic_loss, self.value_optimizer,
                           list(self.value_model.parameters()) + list(self.value_model2.parameters()))

            loss_info.append([
                observation_loss.item(), reward_loss.item(), kl_loss.item(),
                pcont_loss.item() if self.args.pcont else 0,
                actor_loss.item(), critic_loss.item(),
            ])

        # Polyak (soft) update target value networks
        # θ_target = τ * θ_online + (1-τ) * θ_target
        with torch.no_grad():
            for p_online, p_target in zip(
                self.value_model.parameters(), self.target_value_model.parameters()
            ):
                p_target.data.mul_(1 - self.args.polyak).add_(self.args.polyak * p_online.data)
            for p_online, p_target in zip(
                self.value_model2.parameters(), self.target_value_model2.parameters()
            ):
                p_target.data.mul_(1 - self.args.polyak).add_(self.args.polyak * p_online.data)

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

    def save_reconstruction(self, images_dir: str, episode: int, total_episodes: int,
                            n_show: int = 5, max_keep: int = 5) -> None:
        """Sample a replay batch, reconstruct via world model, save real/pred grid.

        Grid layout: real observations (top row) | reconstructions (bottom row), n_show columns.
        Saves ep_NNN.png (zero-padded to total_episodes width) and latest.png.
        Prunes episode images to max_keep randomly selected files.
        """
        self.set_eval_mode()
        with torch.no_grad():
            obs, actions, _, nonterminals = self.D.sample(n_show, self.args.chunk_size)
            init_b = torch.zeros(n_show, self.args.belief_size, device=self.args.device)
            init_s = torch.zeros(n_show, self.args.state_size,  device=self.args.device)
            beliefs, _, _, _, post_states, _, _ = self.transition_model(
                init_s, actions[:-1], init_b,
                bottle(self.encoder, (obs[1:],)),
                nonterminals[:-1],
            )
            recon = bottle(self.observation_model, (beliefs, post_states))
            # Pick the middle timestep from each sequence — representative, not first/last
            mid = self.args.chunk_size // 2
            real = (obs[mid, :n_show] + 0.5).clamp(0, 1)   # [n_show, C, H, W]
            pred = (recon[mid - 1, :n_show] + 0.5).clamp(0, 1)
            grid = make_grid(torch.cat([real, pred], dim=0), nrow=n_show)
        self.set_train_mode()

        ep_str  = str(episode).zfill(len(str(total_episodes)))
        ep_path = os.path.join(images_dir, f'ep_{ep_str}.png')
        save_image(grid, ep_path)
        save_image(grid, os.path.join(images_dir, 'latest.png'))

        imgs = [f for f in os.listdir(images_dir) if f.startswith('ep_') and f.endswith('.png')]
        if len(imgs) > max_keep:
            keep = set(np.random.choice(imgs, max_keep, replace=False))
            for f in imgs:
                if f not in keep:
                    os.remove(os.path.join(images_dir, f))

    def set_train_mode(self):
        for m in (self.transition_model, self.observation_model, self.reward_model,
                  self.encoder, self.actor_model, self.value_model):
            m.train()

    def set_eval_mode(self):
        for m in (self.transition_model, self.observation_model, self.reward_model,
                  self.encoder, self.actor_model, self.value_model):
            m.eval()

    def append_episode(self, obs, actions, rewards, dones) -> None:
        """Append a full episode (numpy arrays) to the replay buffer."""
        for t in range(len(rewards)):
            self.D.append(
                torch.as_tensor(obs[t]),
                actions[t],
                float(rewards[t]),
                bool(dones[t]),
            )

    def save_inference_checkpoint(self, path: str) -> None:
        """Save encoder + transition + actor to a lightweight checkpoint for TFLite export."""
        torch.save({
            'encoder':          self.encoder.cpu().state_dict(),
            'transition_model': self.transition_model.cpu().state_dict(),
            'actor_model':      self.actor_model.cpu().state_dict(),
        }, path)
        self.encoder.to(self.args.device)
        self.transition_model.to(self.args.device)
        self.actor_model.to(self.args.device)

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            'transition_model':  self.transition_model.state_dict(),
            'observation_model': self.observation_model.state_dict(),
            'reward_model':      self.reward_model.state_dict(),
            'encoder':           self.encoder.state_dict(),
            'actor_model':       self.actor_model.state_dict(),
            'value_model':       self.value_model.state_dict(),
            'value_model2':      self.value_model2.state_dict(),
            'world_optimizer':   self.world_optimizer.state_dict(),
            'actor_optimizer':   self.actor_optimizer.state_dict(),
            'value_optimizer':   self.value_optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.args.device)
        self.transition_model.load_state_dict(ckpt['transition_model'])
        self.observation_model.load_state_dict(ckpt['observation_model'])
        self.reward_model.load_state_dict(ckpt['reward_model'])
        self.encoder.load_state_dict(ckpt['encoder'])
        self.actor_model.load_state_dict(ckpt['actor_model'])
        self.value_model.load_state_dict(ckpt['value_model'])
        self.value_model2.load_state_dict(ckpt['value_model2'])

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
