"""
Pull the initial inference.tflite from the server and save it locally.

Run on the PI before drive_physical_tflite.py.
The server runs push_init_model.py simultaneously.

Usage:
  python scripts/pull_init_model.py --server_ip 192.168.0.103
  python scripts/pull_init_model.py --server_ip 192.168.0.103 --output models/inference.tflite
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
parser.add_argument('--output',    default='inference.tflite',
                    help='Where to save the received model (default: inference.tflite)')
args = parser.parse_args()

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.RCVTIMEO, 60_000)   # 60 s timeout
sock.connect(f'tcp://{args.server_ip}:{MODEL_PORT}')

print(f'[Init] Connecting to {args.server_ip}:{MODEL_PORT} ...')
try:
    data    = pickle.loads(sock.recv())
    model_bytes = data['model_bytes']
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(model_bytes)
    print(f'[Init] Saved {len(model_bytes) / 1024:.0f} KB → {args.output}')
except zmq.Again:
    print(f'[Init] Timed out — server did not send a model within 60 s.')
    sys.exit(1)
finally:
    sock.close()
    ctx.term()
