"""
Human-in-the-loop override controller for DonkeyCar training.

Listens for global keypresses using pynput — no window needed.
Watch the sim window and press keys at any time.

Controls:
  = (equals)  — stop: end episode with penalty (car went off track)
  ↑ (up)      — reset: end episode cleanly (clean lap / goal reached)
  ↓ (down)    — quit training

Reward when active:
  +1.0 per step survived
  -1.0 on STOP  (off-track penalty)
   0.0 on RESET (clean lap, neutral)

macOS note: first run may prompt for Accessibility permission in
  System Settings → Privacy & Security → Accessibility.
  Grant it once, then rerun.
"""

import threading


class HumanOverride:
    STOP  = 'stop'
    RESET = 'reset'
    QUIT  = 'quit'

    def __init__(self):
        self._event = None
        self._lock  = threading.Lock()
        self._listener = None
        self._start_listener()

        print('\n' + '─' * 50)
        print('  Human Override ACTIVE')
        print('  Watch the sim window, press keys anytime:')
        print('    =          →  STOP  (car off track)')
        print('    ↑ Up       →  RESET (clean lap)')
        print('    ↓ Down     →  QUIT  (end training)')
        print('─' * 50 + '\n')

    def _start_listener(self):
        from pynput import keyboard

        def on_press(key):
            try:
                ch = key.char.lower() if hasattr(key, 'char') and key.char else None
            except Exception:
                ch = None

            with self._lock:
                if ch == '=':
                    self._event = self.STOP
                    print('\r[HumanOverride] STOP pressed — episode ended with penalty    ')
                elif key == keyboard.Key.up:
                    self._event = self.RESET
                    print('\r[HumanOverride] RESET pressed — clean episode end             ')
                elif key == keyboard.Key.down:
                    self._event = self.QUIT
                    print('\r[HumanOverride] QUIT pressed — stopping training               ')
                    return False  # stops the listener

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def consume_event(self):
        """Return pending event (STOP/RESET/QUIT) and clear it. None if none."""
        with self._lock:
            ev = self._event
            self._event = None
        return ev

    @property
    def should_quit(self):
        with self._lock:
            return self._event == self.QUIT

    def update_frame(self, frame_bgr, stats: dict):
        pass  # no display — user watches the sim window

    def close(self):
        if self._listener:
            self._listener.stop()
