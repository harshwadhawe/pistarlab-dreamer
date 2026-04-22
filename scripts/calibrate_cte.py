"""
CTE calibration — drive manually for 10s and observe CTE.

Controls (hold keys):
  ↑ Up    — throttle forward
  ← Left  — steer left
  → Right — steer right

Usage:
    python scripts/calibrate_cte.py

Requires the sim to already be running on port 9091.
"""

import sys
import time
import numpy as np
from pynput import keyboard

sys.path.insert(0, '.')
from dreamer.config import load_config

DURATION   = 10
THROTTLE   = 0.3
STEER_STEP = 1.0   # full lock left/right

pressed = set()

def on_press(key):
    pressed.add(key)

def on_release(key):
    pressed.discard(key)

def get_action():
    steer    = 0.0
    throttle = 0.0
    if keyboard.Key.right in pressed:
        steer = STEER_STEP
    elif keyboard.Key.left in pressed:
        steer = -STEER_STEP
    if keyboard.Key.up in pressed:
        throttle = THROTTLE
    return np.array([steer, throttle], dtype=np.float32)


def main():
    args = load_config('sim')

    import gymnasium as gym
    import gym_donkeycar  # noqa: F401

    conf = {'host': args.host, 'port': args.port, 'max_cte': 99.0}
    print(f'\nConnecting — {args.env} @ {args.host}:{args.port}')
    env = gym.make(args.env, conf=conf)
    env.reset(seed=args.seed)

    print(f'\nControls: ↑=forward  ←=left  →=right')
    print(f'Monitoring for {DURATION}s — drive to each edge.\n')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    cte_log = []
    start = time.time()
    while time.time() - start < DURATION:
        elapsed = time.time() - start
        action = get_action()
        _, _, _, _, info = env.step(action)
        cte = float(info.get('cte', 0.0))
        cte_log.append(cte)
        bar = '█' * int(abs(cte) / 1.5)
        side = 'R' if cte >= 0 else 'L'
        print(f'\r  {elapsed:4.1f}s  CTE={cte:+7.3f}  {side} {bar:<12}', end='', flush=True)

    listener.stop()
    env.close()

    print(f'\n\n  max CTE (right): {max(cte_log):+.3f}')
    print(f'  min CTE (left):  {min(cte_log):+.3f}')


if __name__ == '__main__':
    main()
