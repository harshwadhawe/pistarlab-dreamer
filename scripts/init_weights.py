"""
Create a random-init Dreamer checkpoint and export it to TFLite.
Use this when training from scratch on real car data (no sim checkpoint).

Usage:
  python scripts/init_weights.py
  python scripts/init_weights.py --channels 3 --output my_init.tflite
"""

import argparse
import subprocess
import sys
import torch

sys.path.insert(0, '.')
from dreamer.models.world_model import TransitionModel, VisualObservationModel, RewardModel, VisualEncoder
from dreamer.models.policy import ActorModel, ValueModel

parser = argparse.ArgumentParser()
parser.add_argument('--channels',      type=int,   default=1,    help='1=grayscale 3=RGB')
parser.add_argument('--belief-size',   type=int,   default=200)
parser.add_argument('--state-size',    type=int,   default=30)
parser.add_argument('--action-size',   type=int,   default=2)
parser.add_argument('--embedding-size',type=int,   default=1024)
parser.add_argument('--hidden-size',   type=int,   default=300)
parser.add_argument('--throttle-base', type=float, default=0.3)
parser.add_argument('--ckpt',          type=str,   default='init_weights.pth')
parser.add_argument('--output',        type=str,   default='inference.tflite')
args = parser.parse_args()

print(f'Creating random-init weights (channels={args.channels})...')

torch.save({
    'transition_model':  TransitionModel(args.belief_size, args.state_size, args.action_size, args.hidden_size, args.embedding_size).state_dict(),
    'observation_model': VisualObservationModel(args.belief_size, args.state_size, args.embedding_size, channels=args.channels).state_dict(),
    'reward_model':      RewardModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
    'encoder':           VisualEncoder(args.embedding_size, channels=args.channels).state_dict(),
    'actor_model':       ActorModel(args.action_size, args.belief_size, args.state_size, args.hidden_size, fix_speed=True, throttle_base=args.throttle_base).state_dict(),
    'value_model':       ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
    'value_model2':      ValueModel(args.belief_size, args.state_size, args.hidden_size).state_dict(),
    'world_optimizer':   {},
    'actor_optimizer':   {},
    'value_optimizer':   {},
}, args.ckpt)
print(f'Saved → {args.ckpt}')

print(f'Exporting TFLite → {args.output}...')
cmd = [
    sys.executable, 'scripts/export_pth_to_tflite.py', args.ckpt,
    '--output',        args.output,
    '--channels',      str(args.channels),
    '--belief-size',   str(args.belief_size),
    '--state-size',    str(args.state_size),
    '--action-size',   str(args.action_size),
    '--embedding-size',str(args.embedding_size),
    '--hidden-size',   str(args.hidden_size),
    '--throttle-base', str(args.throttle_base),
]
result = subprocess.run(cmd, capture_output=True, text=True)
for line in result.stdout.splitlines():
    if any(x in line for x in ['Exported', 'action:', 'new_belief:', 'new_state:', 'Converting']):
        print(line)
if result.returncode != 0:
    print(result.stderr[-400:])
    sys.exit(1)

print(f'\nDone. Copy to car:')
print(f'  scp {args.output} harsh@donkeycar:~/pistarlab-dreamer/{args.output}')
