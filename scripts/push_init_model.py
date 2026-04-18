"""
Export both RGB and grayscale TFLite models and push them to the Pi once via ZMQ.

Run on the SERVER once. The Pi runs pull_init_model.py simultaneously.
After this, the Pi has inference_rgb.tflite and inference_grayscale.tflite and
never needs to run init scripts again — drive_physical_tflite.py auto-selects.

Usage:
  # Random init (first run, no checkpoint):
  python scripts/push_init_model.py

  # Bootstrap from a sim/real checkpoint:
  python scripts/push_init_model.py --models results/real/export_weights_12.pth
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import torch
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dreamer.comms import MODEL_PORT
from dreamer.models.world_model import TransitionModel, VisualObservationModel, RewardModel, VisualEncoder
from dreamer.models.policy import ActorModel, ValueModel

parser = argparse.ArgumentParser()
parser.add_argument('--models',        type=str, default='',
                    help='Path to .pth checkpoint. If empty, random init weights are used.')
parser.add_argument('--bind_ip',       type=str, default='*')
parser.add_argument('--belief-size',   type=int, default=200)
parser.add_argument('--state-size',    type=int, default=30)
parser.add_argument('--action-size',   type=int, default=2)
parser.add_argument('--embedding-size',type=int, default=1024)
parser.add_argument('--hidden-size',   type=int, default=300)
parser.add_argument('--throttle-base', type=float, default=0.3)
args = parser.parse_args()


def _make_ckpt(channels):
    path = tempfile.mktemp(suffix=f'_init_{channels}ch.pth')
    torch.save({
        'transition_model':  TransitionModel(args.belief_size, args.state_size, args.action_size, args.hidden_size, args.embedding_size).state_dict(),
        'observation_model': VisualObservationModel(args.belief_size, args.state_size, args.embedding_size, channels=channels).state_dict(),
        'reward_model':      RewardModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'encoder':           VisualEncoder(args.embedding_size, channels=channels).state_dict(),
        'actor_model':       ActorModel(args.action_size, args.belief_size, args.state_size, args.hidden_size, fix_speed=True, throttle_base=args.throttle_base).state_dict(),
        'value_model':       ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'value_model2':      ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
        'world_optimizer': {}, 'actor_optimizer': {}, 'value_optimizer': {},
    }, path)
    return path


def _export_tflite(ckpt_path, channels):
    out = tempfile.mktemp(suffix=f'_{channels}ch.tflite')
    cmd = [
        sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
        '--output',         out,
        '--channels',       str(channels),
        '--belief-size',    str(args.belief_size),
        '--state-size',     str(args.state_size),
        '--action-size',    str(args.action_size),
        '--embedding-size', str(args.embedding_size),
        '--hidden-size',    str(args.hidden_size),
        '--throttle-base',  str(args.throttle_base),
    ]
    print(f'[Init] Exporting TFLite (channels={channels})...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[Init] Export failed:\n{result.stderr}')
        sys.exit(1)
    print(result.stdout.strip())
    return out


# --- Build both tflite files ---
models = []
for ch, label in [(3, 'rgb'), (1, 'grayscale')]:
    if args.models and os.path.exists(args.models):
        ckpt = args.models
        print(f'[Init] Using checkpoint: {ckpt} (channels={ch})')
    else:
        print(f'[Init] Generating random init weights (channels={ch})...')
        ckpt = _make_ckpt(ch)

    tflite = _export_tflite(ckpt, ch)
    with open(tflite, 'rb') as f:
        model_bytes = f.read()
    models.append((label, model_bytes))
    os.remove(tflite)
    if not args.models:
        os.remove(ckpt)

# --- Push both via ZMQ (sequential) ---
ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.setsockopt(zmq.SNDTIMEO, 60_000)
sock.bind(f'tcp://{args.bind_ip}:{MODEL_PORT}')

print(f'\n[Init] Bound on port {MODEL_PORT}. Waiting for Pi to connect...')
try:
    for label, model_bytes in models:
        payload = pickle.dumps({'label': label, 'model_bytes': model_bytes})
        sock.send(payload)
        print(f'[Init] Pushed inference_{label}.tflite ({len(model_bytes) / 1024:.0f} KB)')
    print('[Init] Both models pushed. Pi is ready.')
except zmq.Again:
    print('[Init] Timed out — Pi did not connect within 60 s.')
    sys.exit(1)
finally:
    sock.close()
    ctx.term()
