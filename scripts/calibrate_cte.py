"""
CTE calibration — drive manually and observe CTE range.

Controls (hold keys):
  ↑ Up    — throttle forward
  ← Left  — steer left
  → Right — steer right
  Esc     — finish and print results

Usage:
    python scripts/calibrate_cte.py

Requires the sim to already be running on port 9091.
"""

import sys
import numpy as np
from pynput import keyboard

sys.path.insert(0, '.')
from dreamer.config import load_config

THROTTLE   = 0.3
STEER_STEP = 1.0

pressed  = set()
finished = False


def on_press(key):
    global finished
    if key == keyboard.Key.esc:
        finished = True
    else:
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
    global finished
    args = load_config('sim')

    import gymnasium as gym
    import gym_donkeycar  # noqa: F401

    conf = {'host': args.host, 'port': args.port, 'max_cte': 99.0}
    print(f'\nConnecting — {args.env} @ {args.host}:{args.port}')
    env = gym.make(args.env, conf=conf)
    env.reset(seed=args.seed)

    print('\nControls: ↑=forward  ←=left  →=right  Esc=done\n')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    cte_log = []
    while not finished:
        action = get_action()
        _, _, terminated, _, info = env.step(action)
        cte = float(info.get('cte', 0.0))
        cte_log.append(cte)
        bar  = '█' * int(abs(cte) / 1.5)
        side = 'R' if cte >= 0 else 'L'
        print(f'\r  CTE={cte:+7.3f}  {side} {bar:<14}  [Esc to finish]', end='', flush=True)
        if terminated:
            env.reset()

    listener.stop()
    env.close()

    if not cte_log:
        print('\nNo data collected.')
        return

    print(f'\n\n  max CTE (right): {max(cte_log):+.3f}')
    print(f'  min CTE (left):  {min(cte_log):+.3f}')
    print()
    print('  Suggested config.toml [sim] values:')
    print(f'    cte_right = {int(abs(max(cte_log))) + 1}')
    print(f'    cte_left  = {int(abs(min(cte_log))) + 1}')


if __name__ == '__main__':
    main()
