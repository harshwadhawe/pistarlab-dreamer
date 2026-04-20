"""
Export both RGB and grayscale TFLite models and serve them to the Pi via HTTP.

Architecture values are read from config.toml [common]. No flags needed
unless you want to override a specific value.

Run on the SERVER once. The Pi runs pull_init_model.py simultaneously.
After this, the Pi has models/inference_rgb.tflite and models/inference_grayscale.tflite
and never needs to run init scripts again — train_real_pi.py auto-selects.

Usage:
  # Random init (first run, no checkpoint):
  python scripts/push_init_model.py

  # Bootstrap from a sim/real checkpoint:
  python scripts/push_init_model.py --models results/real/.../models_50.pth
"""

import argparse
import http.server
import os
import socketserver
import subprocess
import sys
import tempfile
import threading

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dreamer.config import load_config
from dreamer.comms import MODEL_HTTP_PORT
from dreamer.models.world_model import TransitionModel, VisualObservationModel, RewardModel, VisualEncoder
from dreamer.models.policy import ActorModel, ValueModel

cfg = load_config('real')

parser = argparse.ArgumentParser()
parser.add_argument('--models',        type=str,   default='',
                    help='Path to .pth checkpoint. If empty, random init weights are used.')
parser.add_argument('--bind_ip',       type=str,   default='')
# Architecture overrides — defaults come from config.toml [common]
parser.add_argument('--belief-size',   type=int,   default=cfg.belief_size)
parser.add_argument('--state-size',    type=int,   default=cfg.state_size)
parser.add_argument('--action-size',   type=int,   default=cfg.action_size)
parser.add_argument('--embedding-size',type=int,   default=cfg.embedding_size)
parser.add_argument('--hidden-size',   type=int,   default=cfg.hidden_size)
parser.add_argument('--throttle-base', type=float, default=cfg.throttle_base)
args = parser.parse_args()

print(
    f'[Init] Config — belief={args.belief_size} state={args.state_size} '
    f'embed={args.embedding_size} hidden={args.hidden_size} '
    f'throttle={args.throttle_base}'
)


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
models = {}
for ch, label in [(3, 'rgb'), (1, 'grayscale')]:
    if args.models and os.path.exists(args.models):
        ckpt = args.models
        print(f'[Init] Using checkpoint: {ckpt} (channels={ch})')
    else:
        print(f'[Init] Generating random init weights (channels={ch})...')
        ckpt = _make_ckpt(ch)

    tflite = _export_tflite(ckpt, ch)
    with open(tflite, 'rb') as f:
        models[label] = f.read()
    os.remove(tflite)
    if not args.models:
        os.remove(ckpt)

# --- Serve via HTTP ---
_models = models

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        for label, data in _models.items():
            if self.path == f'/model/{label}':
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                print(f'[Init] Served inference_{label}.tflite ({len(data)//1024} KB)')
                return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_): pass


class _ReuseServer(socketserver.TCPServer):
    allow_reuse_address = True


host = args.bind_ip if args.bind_ip else ''
server = _ReuseServer((host, MODEL_HTTP_PORT), Handler)
print(f'\n[Init] Serving on port {MODEL_HTTP_PORT}. Run pull_init_model.py on the Pi...')
print(f'       Ctrl+C to stop once Pi has downloaded both models.')
try:
    server.serve_forever()
except KeyboardInterrupt:
    print('\n[Init] Done.')
