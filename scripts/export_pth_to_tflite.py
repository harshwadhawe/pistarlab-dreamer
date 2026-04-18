"""
Export a Dreamer .pth checkpoint to a single fused TFLite inference model.

The exported model runs one RSSM step end-to-end:
  inputs:  obs [1,C,64,64], prev_belief [1,200], prev_state [1,30], prev_action [1,2]
  outputs: action [1,2], new_belief [1,200], new_state [1,30]

Run in the donkeycar-dreamer conda env (litert_torch is installed there):
  conda activate donkeycar-dreamer
  python scripts/export_pth_to_tflite.py results/.../models_500.pth

Usage:
  python scripts/export_pth_to_tflite.py results/donkey-generated-roads-v0/1/models_500.pth
  python scripts/export_pth_to_tflite.py models_500.pth --channels 3 --output inference.tflite
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch

sys.path.insert(0, '.')
from dreamer.models.world_model import TransitionModel
from dreamer.models.policy import ActorModel


class PlainEncoder(nn.Module):
    """
    Plain nn.Module equivalent of VisualEncoder.
    Required because VisualEncoder inherits jit.ScriptModule which is
    incompatible with torch.export (used internally by litert_torch.convert).
    Weights are loaded from the checkpoint via load_state_dict.
    """

    def __init__(self, embedding_size=1024, channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32,  4, stride=2)
        self.conv2 = nn.Conv2d(32,        64,  4, stride=2)
        self.conv3 = nn.Conv2d(64,       128,  4, stride=2)
        self.conv4 = nn.Conv2d(128,      256,  4, stride=2)
        self.fc    = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 1024)
        return self.fc(x)


class InferenceGraph(nn.Module):
    """
    Fused single-step RSSM inference.

    Stateless — caller tracks belief/state between steps.
    Deterministic: uses posterior mean (no sampling) for stable TFLite export.
    Assumes fix_speed=True: outputs [steering, throttle_base].
    """

    def __init__(self, encoder, transition, actor):
        super().__init__()
        self.encoder    = encoder
        self.transition = transition
        self.actor      = actor

    def forward(self, obs, prev_belief, prev_state, prev_action):
        # --- Encoder ---
        embedding = self.encoder(obs)                         # [1, 1024]

        # --- GRU step ---
        h = F.relu(self.transition.fc_embed_state_action(
            torch.cat([prev_state, prev_action], dim=1)
        ))
        new_belief = self.transition.rnn_norm(
            self.transition.rnn(h, prev_belief)
        )

        # --- Posterior (uses real obs) ---
        ph = F.relu(self.transition.fc_embed_belief_posterior(
            torch.cat([new_belief, embedding], dim=1)
        ))
        post_out = self.transition.fc_state_posterior(ph)
        post_mean, _ = torch.chunk(post_out, 2, dim=1)
        new_state = post_mean                                 # deterministic

        # --- Actor (deterministic mode, fix_speed=True) ---
        ah = self.actor.act_fn(self.actor.fc1(torch.cat([new_belief, new_state], dim=-1)))
        ah = self.actor.act_fn(self.actor.fc2(ah))
        ah = self.actor.act_fn(self.actor.fc3(ah))
        ah = self.actor.act_fn(self.actor.fc4(ah))
        ah = self.actor.fc5(ah)
        raw_mean, _ = torch.chunk(ah, 2, dim=-1)             # steering only
        scaled_mean = self.actor.mean_scale * torch.tanh(raw_mean / self.actor.mean_scale)
        # Transform: AffineTransform(0,2) → Sigmoid → AffineTransform(-1,2)
        steering = 2.0 * torch.sigmoid(2.0 * scaled_mean) - 1.0   # [-1, 1]
        throttle = torch.full_like(steering, self.actor.throttle_base)
        action = torch.cat([steering, throttle], dim=-1)     # [1, 2]

        return action, new_belief, new_state


def build_models(channels, belief_size, state_size, action_size,
                 embedding_size, hidden_size, fix_speed, throttle_base):
    encoder = PlainEncoder(
        embedding_size=embedding_size,
        channels=channels,
    ).eval()

    transition = TransitionModel(
        belief_size=belief_size,
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
    ).eval()

    actor = ActorModel(
        action_size=action_size,
        belief_size=belief_size,
        state_size=state_size,
        hidden_size=hidden_size,
        fix_speed=fix_speed,
        throttle_base=throttle_base,
    ).eval()

    return encoder, transition, actor


def export(checkpoint_path, output_path, channels, belief_size, state_size,
           action_size, embedding_size, hidden_size, fix_speed, throttle_base):

    print(f'Loading checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    encoder, transition, actor = build_models(
        channels, belief_size, state_size, action_size,
        embedding_size, hidden_size, fix_speed, throttle_base,
    )

    encoder.load_state_dict(ckpt['encoder'])
    transition.load_state_dict(ckpt['transition_model'])
    actor.load_state_dict(ckpt['actor_model'])

    graph = InferenceGraph(encoder, transition, actor).eval().cpu()

    dummy = (
        torch.zeros(1, channels, 64, 64),
        torch.zeros(1, belief_size),
        torch.zeros(1, state_size),
        torch.zeros(1, action_size),
    )

    print('Converting to TFLite (may take a minute)...')
    with torch.no_grad():
        edge_model = litert_torch.convert(graph, dummy)

    edge_model.export(output_path)
    print(f'Exported → {output_path}')

    # Quick shape check
    action_out, belief_out, state_out = graph(*dummy)
    print(f'  action:     {tuple(action_out.shape)}')
    print(f'  new_belief: {tuple(belief_out.shape)}')
    print(f'  new_state:  {tuple(state_out.shape)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint',      help='Path to models_N.pth checkpoint')
    parser.add_argument('--output',        default='inference.tflite')
    parser.add_argument('--channels',      type=int,   default=1,    help='1=grayscale, 3=RGB')
    parser.add_argument('--belief-size',   type=int,   default=200)
    parser.add_argument('--state-size',    type=int,   default=30)
    parser.add_argument('--action-size',   type=int,   default=2)
    parser.add_argument('--embedding-size',type=int,   default=1024)
    parser.add_argument('--hidden-size',   type=int,   default=300)
    parser.add_argument('--fix-speed',     action='store_true', default=True)
    parser.add_argument('--throttle-base', type=float, default=0.3)
    args = parser.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        channels=args.channels,
        belief_size=args.belief_size,
        state_size=args.state_size,
        action_size=args.action_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        fix_speed=args.fix_speed,
        throttle_base=args.throttle_base,
    )
