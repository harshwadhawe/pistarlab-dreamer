"""
Quick PS4 controller test on Pi5.
Press each button to verify detection. Ctrl+C to exit.

Run on car:
  python scripts/test_ps4.py
"""

import sys, time
sys.path.insert(0, '.')
from dreamer.envs.ps4_override import PS4Override

ps4 = PS4Override()
print('Press buttons on PS4 controller. Ctrl+C to exit.\n')

try:
    while True:
        ev = ps4.consume_event()
        if ev:
            print(f'  Event: {ev}  |  paused={ps4.is_paused}')
            if ev == PS4Override.QUIT:
                print('QUIT received — exiting.')
                break
        time.sleep(0.05)
except KeyboardInterrupt:
    print('\nExiting.')
finally:
    ps4.stop()
