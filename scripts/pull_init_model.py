"""
Pull both inference_rgb.tflite and inference_grayscale.tflite from the server.

Run on the PI once. After this, drive_physical_tflite.py auto-selects the
correct model based on --channels — no init scripts needed on future runs.

Usage:
  python scripts/pull_init_model.py --server_ip 192.168.0.103
"""

import argparse
import os
import pickle
import sys

import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dreamer.comms import MODEL_PORT

parser = argparse.ArgumentParser()
parser.add_argument('--server_ip', required=True, help='Server IP address')
parser.add_argument('--output_dir', default='models', help='Directory to save models (default: models/)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.RCVTIMEO, 120_000)   # 2 min — export takes time
sock.connect(f'tcp://{args.server_ip}:{MODEL_PORT}')

print(f'[Init] Connecting to {args.server_ip}:{MODEL_PORT} ...')
try:
    for _ in range(2):   # expect rgb + grayscale
        data        = pickle.loads(sock.recv())
        label       = data['label']          # 'rgb' or 'grayscale'
        model_bytes = data['model_bytes']
        out_path    = os.path.join(args.output_dir, f'inference_{label}.tflite')
        with open(out_path, 'wb') as f:
            f.write(model_bytes)
        print(f'[Init] Saved inference_{label}.tflite ({len(model_bytes) / 1024:.0f} KB) → {out_path}')
    print('[Init] Done. Run drive_physical_tflite.py with --channels 3 (RGB) or --channels 1 (grayscale).')
except zmq.Again:
    print('[Init] Timed out — server did not send models within 120 s.')
    sys.exit(1)
finally:
    sock.close()
    ctx.term()
