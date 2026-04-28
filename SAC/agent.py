from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F


class SACAgent:
    """
    SAC with frozen VAE encoder. Auto-temperature. Twin Q critics with soft target update.

    Key difference from Dreamer: Q(z,a) Bellman backup on real transitions,
    no imagination rollouts, no temporal memory (stateless z per frame).
    """

    def __init__(self, vae, actor, critic, args):
        self.vae    = vae
        self.actor  = actor
        self.critic = critic
        self.args   = args

        self.critic_target = deepcopy(critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt  = torch.optim.Adam(actor.parameters(),  lr=args.actor_lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=args.value_lr)

        self.log_alpha = torch.tensor([float(np.log(0.1))], requires_grad=True, device=args.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.actor_lr)
        self.target_entropy = float(-args.action_size)   # -2 for (steer, throttle)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def encode(self, obs):
        """Encode obs batch → z using VAE mean (no grad, frozen encoder)."""
        with torch.no_grad():
            mean, _ = self.vae.encode(obs)
        return mean

    def update(self, replay, batch_size):
        obs, next_obs, actions, rewards, dones = replay.sample(batch_size)

        z      = self.encode(obs)
        z_next = self.encode(next_obs)

        # ── Critic (Bellman backup) ────────────────────────────────────────
        with torch.no_grad():
            next_action, next_logp = self.actor(z_next)
            q_next   = self.critic_target.min_q(z_next, next_action) - self.alpha * next_logp
            q_target = rewards + self.args.discount * (~dones) * q_next

        q1, q2 = self.critic(z, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor (maximize Q - α·log π) ──────────────────────────────────
        action, logp = self.actor(z)
        actor_loss = (self.alpha.detach() * logp - self.critic.min_q(z, action)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Temperature (auto-tune entropy target) ─────────────────────────
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft target update ─────────────────────────────────────────────
        tau = self.args.polyak
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - tau).add_(p.data * tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss':  actor_loss.item(),
            'alpha':       self.alpha.item(),
        }
