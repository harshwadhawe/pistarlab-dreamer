"""
Export a TFLite model and push it to the Pi once via ZMQ.

Run on the SERVER before starting train_real.py and drive_physical_tflite.py.
The Pi runs pull_init_model.py simultaneously to receive it.

Usage:
  # From latest sim checkpoint, grayscale:
  python scripts/push_init_model.py --models results/real/export_weights_12.pth --channels 1

  # RGB, random init:
  python scripts/push_init_model.py --channels 3
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import zmq

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dreamer.comms import MODEL_PORT
from dreamer.models.world_model import TransitionModel, VisualObservationModel, RewardModel, VisualEncoder
from dreamer.models.policy import ActorModel, ValueModel

parser = argparse.ArgumentParser()
parser.add_argument('--models',        type=str, default='',
                    help='Path to .pth checkpoint. If empty, random init weights are used.')
parser.add_argument('--channels',      type=int, default=1, help='1=grayscale  3=RGB')
parser.add_argument('--bind_ip',       type=str, default='*')
parser.add_argument('--belief-size',   type=int, default=200)
parser.add_argument('--state-size',    type=int, default=30)
parser.add_argument('--action-size',   type=int, default=2)
parser.add_argument('--embedding-size',type=int, default=1024)
parser.add_argument('--hidden-size',   type=int, default=300)
parser.add_argument('--throttle-base', type=float, default=0.3)
args = parser.parse_args()

# --- Resolve or generate checkpoint ---
if args.models and os.path.exists(args.models):
    ckpt_path = args.models
    print(f'[Init] Using checkpoint: {ckpt_path}')
else:
    print(f'[Init] No checkpoint — generating random init weights (channels={args.channels})...')
    ckpt_path = tempfile.mktemp(suffix='_init_weights.pth')
    torch.save({
        'transition_model':  TransitionModel(args.belief_size, args.state_size, args.action_size, args.hidden_size, args.embedding_size).state_dict(),
        'observation_model': VisualObservationModel(args.belief_size, args.state_size, args.embedding_size, channels=args.channels).state_dict(),
        'reward_model':      RewardModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'encoder':           VisualEncoder(args.embedding_size, channels=args.channels).state_dict(),
        'actor_model':       ActorModel(args.action_size, args.belief_size, args.state_size, args.hidden_size, fix_speed=True, throttle_base=args.throttle_base).state_dict(),
        'value_model':       ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'value_model2':      ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'world_optimizer': {}, 'actor_optimizer': {}, 'value_optimizer': {},
    }, ckpt_path)
    print(f'[Init] Random weights saved → {ckpt_path}')

# --- Export TFLite ---
tflite_path = tempfile.mktemp(suffix='_init.tflite')
cmd = [
    sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
    '--output',         tflite_path,
    '--channels',       str(args.channels),
    '--belief-size',    str(args.belief_size),
    '--state-size',     str(args.state_size),
    '--action-size',    str(args.action_size),
    '--embedding-size', str(args.embedding_size),
    '--hidden-size',    str(args.hidden_size),
    '--throttle-base',  str(args.throttle_base),
]
print(f'[Init] Exporting TFLite (channels={args.channels})...')
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f'[Init] Export failed:\n{result.stderr}')
    sys.exit(1)
print(result.stdout.strip())

# --- Push via ZMQ ---
with open(tflite_path, 'rb') as f:
    model_bytes = f.read()

payload = pickle.dumps({'model_bytes': model_bytes, 'step': 0})

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.setsockopt(zmq.SNDTIMEO, 30_000)
sock.bind(f'tcp://{args.bind_ip}:{MODEL_PORT}')

print(f'[Init] Bound on port {MODEL_PORT}. Waiting for Pi to connect...')
try:
    sock.send(payload)
    print(f'[Init] Model pushed ({len(model_bytes) / 1024:.0f} KB). Pi is ready.')
except zmq.Again:
    print(f'[Init] Timed out — Pi did not connect within 30 s.')
    sys.exit(1)
finally:
    sock.close()
    ctx.term()
    if os.path.exists(tflite_path):
        os.remove(tflite_path)
