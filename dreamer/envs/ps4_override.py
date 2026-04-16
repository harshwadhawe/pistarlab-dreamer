"""
PS4 controller override for Pi5 real-world driving.
Uses evdev — reads directly from /dev/input/event*, no root needed (input group required).

Button mapping:
  ○ Circle   → STOP  (episode end, reward -1)
  × Cross    → RESET (clean lap end, reward 0)
  △ Triangle → QUIT  (end session)
  □ Square   → PAUSE (toggle zero throttle)
"""

import threading
import evdev
from evdev import InputDevice, categorize, ecodes


# Standard Linux gamepad button codes
_BTN_CIRCLE   = 305   # ○
_BTN_CROSS    = 304   # ×
_BTN_TRIANGLE = 308   # △
_BTN_SQUARE   = 307   # □


def _find_controller():
    for path in evdev.list_devices():
        dev = InputDevice(path)
        if 'Wireless Controller' in dev.name or 'DUALSHOCK' in dev.name.upper():
            return dev
    return None


class PS4Override:
    STOP  = 'stop'
    RESET = 'reset'
    QUIT  = 'quit'

    def __init__(self):
        self._event  = None
        self._paused = False
        self._lock   = threading.Lock()
        self._active = True

        dev = _find_controller()
        if dev is None:
            raise RuntimeError(
                'PS4 controller not found. '
                'Check it is paired and user is in the input group.'
            )
        print(f'[PS4] Found: {dev.name} ({dev.path})')
        print('[PS4] ○=STOP  ×=RESET  △=QUIT  □=PAUSE')

        t = threading.Thread(target=self._poll_loop, args=(dev,), daemon=True)
        t.start()

    def _poll_loop(self, dev):
        try:
            for event in dev.read_loop():
                if not self._active:
                    break
                if event.type != ecodes.EV_KEY:
                    continue
                key = categorize(event)
                if key.keystate != 1:   # only press, not release
                    continue
                if key.scancode == _BTN_SQUARE:
                    with self._lock:
                        self._paused = not self._paused
                    print(f'\r[PS4] {"PAUSED" if self._paused else "RESUMED"}   ')
                elif key.scancode == _BTN_CIRCLE:
                    with self._lock:
                        self._event = self.STOP
                    print('\r[PS4] STOP   ')
                elif key.scancode == _BTN_CROSS:
                    with self._lock:
                        self._event = self.RESET
                    print('\r[PS4] RESET  ')
                elif key.scancode == _BTN_TRIANGLE:
                    with self._lock:
                        self._event = self.QUIT
                    print('\r[PS4] QUIT   ')
        except Exception as e:
            print(f'[PS4] Error: {e}')

    def consume_event(self):
        with self._lock:
            ev, self._event = self._event, None
        return ev

    @property
    def is_paused(self):
        with self._lock:
            return self._paused

    @property
    def should_quit(self):
        with self._lock:
            return self._event == self.QUIT

    def stop(self):
        self._active = False
