"""
PS4 controller override for Pi5 real-world driving.
Uses the `inputs` library (Linux evdev, no root needed).

Button mapping:
  ○ Circle   → STOP  (episode end, reward -1)
  × Cross    → RESET (clean lap end, reward 0)
  △ Triangle → QUIT  (end session)
  □ Square   → PAUSE (toggle zero throttle)

Same consume_event() / should_quit interface as HumanOverride.
"""

import threading


class PS4Override:
    STOP  = 'stop'
    RESET = 'reset'
    QUIT  = 'quit'

    _BTN_MAP = {
        ('BTN_EAST',  1): 'stop',   # ○ Circle
        ('BTN_SOUTH', 1): 'reset',  # × Cross
        ('BTN_NORTH', 1): 'quit',   # △ Triangle
    }

    def __init__(self):
        self._event  = None
        self._paused = False
        self._lock   = threading.Lock()
        self._active = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        print('[PS4] Listening — ○=STOP  ×=RESET  △=QUIT  □=PAUSE')

    def _poll_loop(self):
        from inputs import get_gamepad
        while self._active:
            try:
                for ev in get_gamepad():
                    key = (ev.code, ev.state)
                    if key == ('BTN_WEST', 1):          # □ Square — toggle pause
                        with self._lock:
                            self._paused = not self._paused
                        print(f'\r[PS4] {"PAUSED" if self._paused else "RESUMED"}   ')
                    elif key in self._BTN_MAP:
                        with self._lock:
                            self._event = self._BTN_MAP[key]
                        print(f'\r[PS4] {self._event.upper()}   ')
            except Exception as e:
                print(f'[PS4] Controller error: {e}')
                break

    def consume_event(self):
        """Return and clear the current event, or None."""
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
