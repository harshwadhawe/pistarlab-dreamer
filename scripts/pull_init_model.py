"""
Download both inference_rgb.tflite and inference_grayscale.tflite from the server.

Run on the PI once. After this, drive_physical_tflite.py auto-selects the
correct model based on --grayscale flag — no init scripts needed on future runs.

Usage:
  python scripts/pull_init_model.py --server_ip 192.168.0.103
"""

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dreamer.comms import MODEL_HTTP_PORT

parser = argparse.ArgumentParser()
parser.add_argument('--server_ip',  required=True, help='Server IP address')
parser.add_argument('--output_dir', default='models', help='Directory to save models (default: models/)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
base = f'http://{args.server_ip}:{MODEL_HTTP_PORT}'

for label in ('rgb', 'grayscale'):
    url      = f'{base}/model/{label}'
    out_path = os.path.join(args.output_dir, f'inference_{label}.tflite')
    print(f'[Init] Downloading inference_{label}.tflite ...')
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        print(f'[Init] Saved {size // 1024} KB → {out_path}')
    except Exception as e:
        print(f'[Init] Failed to download {label}: {e}')
        sys.exit(1)

print('\n[Init] Done. You can now Ctrl+C the server and start train_real.py.')
