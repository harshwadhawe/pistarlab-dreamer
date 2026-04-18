"""
Communication layer — shared by server and car.

Experience transport (car → server): ZMQ PUSH/PULL on port 5555.
Model transport    (server → car):   HTTP on port 5557.

HTTP replaced ZMQ for model delivery — HTTP is 3× faster than ZMQ for the
~6.5 MB TFLite payload on slow WiFi (50s vs 172s in testing).

Server runs a background HTTP server; car polls GET /step (lightweight) and
downloads GET /model only when the step number has advanced.
Download runs in a background thread so the car's control loop stays responsive.

Discard protocol: car sends a lightweight discard notification instead of full
experience when the operator discards an episode, preventing a server deadlock.
"""

import http.server
import pickle
import socketserver
import threading
import urllib.request

import numpy as np
import zmq

EXPERIENCE_PORT  = 5555
MODEL_HTTP_PORT  = 5557


# ---------------------------------------------------------------------------
# Null-object implementations — used when running without a server
# ---------------------------------------------------------------------------

class NoopSender:
    def send(self, **_): pass
    def send_discard(self, episode_num: int): pass


class NoopSubscriber:
    def poll(self): return None


def make_comms(server_ip: str):
    """Return (sender, subscriber) — real pair or no-ops if server_ip is empty."""
    if not server_ip:
        return NoopSender(), NoopSubscriber()
    return ExperienceSender(server_ip), ModelClient(server_ip)


# ---------------------------------------------------------------------------
# Experience transport  (car → server)   ZMQ PUSH/PULL — unchanged
# ---------------------------------------------------------------------------

class ExperienceSender:
    """Car — pushes one episode (or discard notification) to the server."""

    def __init__(self, server_ip: str, port: int = EXPERIENCE_PORT):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.connect(f'tcp://{server_ip}:{port}')
        print(f'[Comms] ExperienceSender → tcp://{server_ip}:{port}')

    def send(self, obs: np.ndarray, actions: np.ndarray,
             rewards: np.ndarray, dones: np.ndarray, meta: dict):
        payload = {
            'obs':       obs,
            'actions':   actions,
            'rewards':   rewards,
            'dones':     dones,
            'discarded': False,
            **meta,
        }
        self.sock.send(pickle.dumps(payload))

    def send_discard(self, episode_num: int):
        """Notify server that this episode was discarded — no experience to train on."""
        self.sock.send(pickle.dumps({'discarded': True, 'episode_num': episode_num}))


class ExperienceReceiver:
    """Server — blocks until an episode or discard notification arrives from the car."""

    def __init__(self, bind_ip: str = '*', port: int = EXPERIENCE_PORT):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PULL)
        self.sock.bind(f'tcp://{bind_ip}:{port}')
        print(f'[Comms] ExperienceReceiver bound on tcp://{bind_ip}:{port}')

    def recv(self) -> dict:
        return pickle.loads(self.sock.recv())


# ---------------------------------------------------------------------------
# Model transport  (server → car)   HTTP — replaces ZMQ
# ---------------------------------------------------------------------------

class _ModelHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler serving /step and /model endpoints."""

    def do_GET(self):
        if self.path == '/step':
            with self.server._lock:
                body = str(self.server._step).encode()
            self._respond(200, 'text/plain', body)

        elif self.path == '/model':
            with self.server._lock:
                data = self.server._model_bytes
                step = self.server._step
            if data is None:
                self._respond(503, 'text/plain', b'not ready')
                return
            headers = {'X-Step': str(step), 'Content-Type': 'application/octet-stream'}
            self._respond(200, 'application/octet-stream', data, extra=headers)

        else:
            self._respond(404, 'text/plain', b'not found')

    def _respond(self, code, content_type, body, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass   # suppress per-request logs


class _ReuseServer(socketserver.TCPServer):
    allow_reuse_address = True


class ModelPublisher:
    """Server — serves the latest TFLite model over HTTP in a background thread.

    Replaces ZMQ PUSH socket. Car polls /step and downloads /model via HTTP,
    which is 3× faster than ZMQ for large payloads on slow WiFi.
    """

    def __init__(self, bind_ip: str = '*', port: int = MODEL_HTTP_PORT):
        self._model_bytes = None
        self._step        = -1
        self._lock        = threading.Lock()

        host = '' if bind_ip == '*' else bind_ip
        self._server = _ReuseServer((host, port), _ModelHTTPHandler)
        self._server._lock        = self._lock
        self._server._model_bytes = None
        self._server._step        = -1

        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        print(f'[Comms] ModelPublisher HTTP server on port {port}')

    def publish(self, tflite_path: str, step: int):
        with open(tflite_path, 'rb') as f:
            model_bytes = f.read()
        with self._lock:
            self._server._model_bytes = model_bytes
            self._server._step        = step
        print(f'[Comms] Model ready — step {step}, {len(model_bytes) // 1024} KB')


class ModelClient:
    """Car — polls the server HTTP endpoint for updated model bytes.

    /step is a tiny text response checked every poll cycle.
    /model (~6.5 MB) is downloaded in a background thread only when step advances,
    keeping the car's control loop non-blocking during the transfer.
    """

    def __init__(self, server_ip: str, port: int = MODEL_HTTP_PORT):
        self._base        = f'http://{server_ip}:{port}'
        self._last_step   = -1
        self._downloading = False
        self._pending     = None   # set by background thread when download completes
        self._lock        = threading.Lock()
        print(f'[Comms] ModelClient → {self._base}')

    def poll(self) -> dict | None:
        """Non-blocking. Returns dict(model_bytes, step) when new model ready, else None."""
        # Return completed download if available
        with self._lock:
            if self._pending is not None:
                result, self._pending = self._pending, None
                return result

        # Don't start another download while one is running
        if self._downloading:
            return None

        # Lightweight step check
        try:
            with urllib.request.urlopen(f'{self._base}/step', timeout=2) as r:
                step = int(r.read().decode())
        except Exception:
            return None

        if step <= self._last_step:
            return None

        # New model available — download in background
        self._downloading = True
        threading.Thread(target=self._download, args=(step,), daemon=True).start()
        return None

    def _download(self, step: int):
        try:
            with urllib.request.urlopen(f'{self._base}/model', timeout=300) as r:
                model_bytes = r.read()
            with self._lock:
                self._pending   = {'model_bytes': model_bytes, 'step': step}
                self._last_step = step
            print(f'\n[Comms] Model downloaded — step {step}, {len(model_bytes) // 1024} KB')
        except Exception as e:
            print(f'\n[Comms] Model download failed: {e}')
        finally:
            self._downloading = False
