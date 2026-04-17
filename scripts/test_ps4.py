"""
Quick PS4 controller test on Pi5.
Press each button to verify detection. Ctrl+C to exit.

Run on car:
  python scripts/test_ps4.py
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dreamer.envs.controller import EpisodeController

ctrl = EpisodeController.from_gamepad()
print('Press buttons on PS4 controller. Ctrl+C to exit.\n')

try:
    while True:
        ev = ctrl.consume_event()
        if ev:
            print(f'  Event: {ev}')
        time.sleep(0.05)
except KeyboardInterrupt:
    print('\nExiting.')
finally:
    ctrl.close()
