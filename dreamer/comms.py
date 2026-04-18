"""
ZMQ communication layer — shared by server and car.

Car (Pi5) side: no PyTorch needed, only pyzmq + numpy.
Server side:    same classes, just different socket roles.

Ports:
  5555 — experience  (Car PUSH  → Server PULL)
  5556 — model       (Server PUSH → Car PULL, CONFLATE=1 keeps only latest)

After each episode the car halts (zero throttle) and blocks on port 5556
until a new model arrives. The server always exports+publishes after every
training episode, so the car unblocks as soon as export finishes.

Model transport uses PUSH/PULL (not PUB/SUB) so messages are queued until
the receiver connects — avoids the slow-joiner race where a PUB fires before
the SUB has completed its TCP handshake. CONFLATE=1 on the PULL socket ensures
only the latest model is kept, discarding stale queued weights.

Discard protocol: when the car operator discards an episode, the car sends a
lightweight discard notification instead of full experience. The server skips
training for that episode and waits for the next one, avoiding a deadlock where
the server blocks on recv() indefinitely.
"""

import pickle
import zlib

import numpy as np
import zmq

EXPERIENCE_PORT = 5555
MODEL_PORT      = 5556


# ---------------------------------------------------------------------------
# Null-object implementations — used when running without a server
# ---------------------------------------------------------------------------

class NoopSender:
    def send(self, **_): pass
    def send_discard(self, episode_num: int): pass


class NoopSubscriber:
    def poll(self): return None


def make_comms(server_ip: str):
    """Return (sender, subscriber) — real ZMQ pair or no-ops if server_ip is empty."""
    if not server_ip:
        return NoopSender(), NoopSubscriber()
    return ExperienceSender(server_ip), ModelSubscriber(server_ip)


# ---------------------------------------------------------------------------
# Experience transport  (car → server)
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
        """
        obs:     [T, C, 64, 64]  float32  [-0.5, 0.5]
        actions: [T, action_size] float32
        rewards: [T]              float32
        dones:   [T]              bool
        meta:    {'episode_num': int, 'steps': int}
        """
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
        payload = {'discarded': True, 'episode_num': episode_num}
        self.sock.send(pickle.dumps(payload))


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
# Model transport  (server → car)
# ---------------------------------------------------------------------------

class ModelPublisher:
    """Server — pushes updated .tflite bytes to the car after each export.

    Uses PUSH socket so messages are queued until the car connects (no slow-joiner
    loss). Car uses CONFLATE=1 so only the latest model is kept.
    """

    _BUF = 8 * 1024 * 1024   # 8 MB socket buffer — needed for large TFLite payloads

    def __init__(self, bind_ip: str = '*', port: int = MODEL_PORT):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDTIMEO, 300_000)  # 5 min — slow WiFi needs time
        self.sock.setsockopt(zmq.SNDBUF,   self._BUF)
        self.sock.bind(f'tcp://{bind_ip}:{port}')
        print(f'[Comms] ModelPublisher bound on tcp://{bind_ip}:{port}')

    def publish(self, tflite_path: str, step: int):
        with open(tflite_path, 'rb') as f:
            model_bytes = f.read()
        compressed = zlib.compress(model_bytes, level=1)  # level=1: fast, ~50% smaller
        payload = pickle.dumps({'model_bytes': compressed, 'step': step, 'compressed': True})
        try:
            self.sock.send(payload)
            print(f'[Comms] Published model — step {step}, '
                  f'{len(model_bytes)//1024} KB → {len(compressed)//1024} KB compressed')
        except zmq.Again:
            print(f'[Comms] WARNING: model publish timed out — '
                  f'car not connected on port {MODEL_PORT}? Skipping.')


class ModelSubscriber:
    """Car — non-blocking poll for updated model bytes from the server."""

    _BUF = 8 * 1024 * 1024   # 8 MB socket buffer

    def __init__(self, server_ip: str, port: int = MODEL_PORT):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVBUF, self._BUF)
        self.sock.connect(f'tcp://{server_ip}:{port}')
        print(f'[Comms] ModelSubscriber ← tcp://{server_ip}:{port}')

    def poll(self) -> dict | None:
        """Returns dict with model_bytes + step if available, else None."""
        try:
            data = pickle.loads(self.sock.recv(zmq.NOBLOCK))
            if data.get('compressed'):
                data['model_bytes'] = zlib.decompress(data['model_bytes'])
            return data
        except zmq.Again:
            return None
