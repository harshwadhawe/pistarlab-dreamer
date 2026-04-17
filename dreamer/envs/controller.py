"""
Episode controller — unified interface for human overrides in sim and real-world training.

Two backends share the same controller interface:
  KeyboardBackend  — arrow keys    (sim, via pynput)
  GamepadBackend   — PS4 buttons   (Pi5, via evdev)

Usage:
  ctrl = EpisodeController.from_keyboard()   # sim
  ctrl = EpisodeController.from_gamepad()    # real car

  ev                        = ctrl.consume_event()
  reward, terminated, discard = ctrl.event_to_outcome(ev)

event_to_outcome reward table:
  STOP    → -1.0  (off track, penalise)
  RESET   → +1.0  (clean lap, same as a survived step)
  DISCARD →  0.0  (episode erased — steps never reach trainer)
  START   →  1.0  (no-event, survived step — START is consumed at episode boundary)
  None    → +1.0  (no event, survived step)
"""

import queue
import threading


class EpisodeController:
    STOP    = 'stop'
    RESET   = 'reset'
    DISCARD = 'discard'
    START   = 'start'

    def __init__(self, backend):
        self._queue   = queue.Queue(maxsize=10)
        self._backend = backend
        backend._attach(self)

    # --- Public API (same for both backends) ---

    def consume_event(self):
        """Return and remove the oldest pending event, or None if none."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @staticmethod
    def event_to_outcome(ev):
        """Map an override event → (reward, terminated, discard_requested)."""
        if ev == EpisodeController.STOP:
            return -1.0, True,  False
        if ev == EpisodeController.RESET:
            return  1.0, True,  False
        if ev == EpisodeController.DISCARD:
            return  0.0, True,  True
        return 1.0, False, False  # None or START — survived step

    def close(self):
        self._backend.close()

    @property
    def should_quit(self):
        return False  # QUIT removed — use Ctrl+C to stop

    # --- Internal: called only by backends ---

    def _push(self, event):
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass  # drop oldest implicitly — maxsize prevents unbounded growth

    # --- Factories ---

    @classmethod
    def from_keyboard(cls):
        """Create a controller driven by arrow keys (sim training)."""
        return cls(KeyboardBackend())

    @classmethod
    def from_gamepad(cls):
        """Create a controller driven by a PS4 gamepad (real-world Pi5)."""
        return cls(GamepadBackend())


# ---------------------------------------------------------------------------
# Keyboard backend — pynput, no window needed
# ---------------------------------------------------------------------------

class KeyboardBackend:
    """
    Arrow key bindings:
      ↑ Up    → RESET
      ↓ Down  → DISCARD
      ← Left  → STOP
      → Right → START
    """

    def __init__(self):
        self._ctrl     = None
        self._listener = None

    def _attach(self, ctrl):
        self._ctrl = ctrl
        self._start_listener()
        print('\n' + '─' * 50)
        print('  Episode Controller — Keyboard')
        print('    ↑ Up    →  RESET   (clean lap, +1)')
        print('    ↓ Down  →  DISCARD (erase episode)')
        print('    ← Left  →  STOP    (off-track, -1)')
        print('    → Right →  START   (begin next episode)')
        print('  Ctrl+C to stop training.')
        print('─' * 50 + '\n')

    def _start_listener(self):
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.up:
                self._ctrl._push(EpisodeController.RESET)
                print('\r[Ctrl] RESET   ')
            elif key == keyboard.Key.down:
                self._ctrl._push(EpisodeController.DISCARD)
                print('\r[Ctrl] DISCARD ')
            elif key == keyboard.Key.left:
                self._ctrl._push(EpisodeController.STOP)
                print('\r[Ctrl] STOP    ')
            elif key == keyboard.Key.right:
                self._ctrl._push(EpisodeController.START)
                print('\r[Ctrl] START   ')

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def close(self):
        if self._listener:
            self._listener.stop()


# ---------------------------------------------------------------------------
# Gamepad backend — evdev, PS4 / DualSense (Pi5 only)
# ---------------------------------------------------------------------------

_BTN_CIRCLE = 305   # ○
_BTN_CROSS  = 304   # ×
_BTN_SQUARE = 308   # □
_BTN_R1     = 311   # R1


class GamepadBackend:
    """
    PS4 button bindings:
      ○ Circle → STOP
      × Cross  → RESET
      □ Square → DISCARD
      R1       → START
    """

    def __init__(self):
        self._ctrl   = None
        self._active = True

    def _attach(self, ctrl):
        self._ctrl = ctrl
        import evdev
        dev = self._find_controller(evdev)
        if dev is None:
            raise RuntimeError(
                'PS4/DualSense controller not found. '
                'Check it is paired and user is in the input group.'
            )
        print(f'[Ctrl] Found: {dev.name} ({dev.path})')
        print('[Ctrl] ○=STOP  ×=RESET  □=DISCARD  R1=START')

        t = threading.Thread(target=self._poll_loop, args=(dev,), daemon=True)
        t.start()

    def _find_controller(self, evdev):
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            if dev.name in ('Wireless Controller', 'DualSense Wireless Controller') \
                    or dev.name.upper() == 'DUALSHOCK 4 WIRELESS CONTROLLER':
                return dev
        return None

    def _poll_loop(self, dev):
        from evdev import categorize, ecodes
        try:
            for event in dev.read_loop():
                if not self._active:
                    break
                if event.type != ecodes.EV_KEY:
                    continue
                key = categorize(event)
                if key.keystate != 1:   # press only, not release
                    continue
                if key.scancode == _BTN_CIRCLE:
                    self._ctrl._push(EpisodeController.STOP)
                    print('\r[Ctrl] STOP    ')
                elif key.scancode == _BTN_CROSS:
                    self._ctrl._push(EpisodeController.RESET)
                    print('\r[Ctrl] RESET   ')
                elif key.scancode == _BTN_SQUARE:
                    self._ctrl._push(EpisodeController.DISCARD)
                    print('\r[Ctrl] DISCARD ')
                elif key.scancode == _BTN_R1:
                    self._ctrl._push(EpisodeController.START)
                    print('\r[Ctrl] START   ')
        except Exception as e:
            print(f'[Ctrl] Gamepad error: {e}')

    def close(self):
        self._active = False
