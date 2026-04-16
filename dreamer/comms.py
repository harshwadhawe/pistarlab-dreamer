"""
ZMQ communication layer — shared by server and car.

Car (Pi5) side: no PyTorch needed, only pyzmq + numpy.
Server side:    same classes, just different socket roles.

Ports:
  5555 — experience  (Car PUSH  → Server PULL)
  5556 — model       (Server PUB → Car SUB)

After each episode the car halts (zero throttle) and blocks on port 5556
until a new model arrives. The server always exports+publishes after every
episode (background thread), so the car unblocks as soon as export finishes.
"""

import pickle
import numpy as np
import zmq

EXPERIENCE_PORT = 5555
MODEL_PORT      = 5556


# ---------------------------------------------------------------------------
# Experience transport  (car → server)
# ---------------------------------------------------------------------------

class ExperienceSender:
    """Car — pushes one episode to the server at episode end."""

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
            'obs':     obs,
            'actions': actions,
            'rewards': rewards,
            'dones':   dones,
            **meta,
        }
        self.sock.send(pickle.dumps(payload))


class ExperienceReceiver:
    """Server — blocks until an episode arrives from the car."""

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
    """Server — broadcasts updated .tflite bytes after each export."""

    def __init__(self, bind_ip: str = '*', port: int = MODEL_PORT):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(f'tcp://{bind_ip}:{port}')
        print(f'[Comms] ModelPublisher bound on tcp://{bind_ip}:{port}')

    def publish(self, tflite_path: str, step: int):
        with open(tflite_path, 'rb') as f:
            model_bytes = f.read()
        payload = pickle.dumps({'model_bytes': model_bytes, 'step': step})
        self.sock.send_multipart([b'model', payload])
        print(f'[Comms] Published model — step {step}, '
              f'size {len(model_bytes) / 1024:.0f} KB')


class ModelSubscriber:
    """Car — non-blocking poll for updated model bytes from the server."""

    def __init__(self, server_ip: str, port: int = MODEL_PORT):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect(f'tcp://{server_ip}:{port}')
        self.sock.setsockopt(zmq.SUBSCRIBE, b'model')
        print(f'[Comms] ModelSubscriber ← tcp://{server_ip}:{port}')

    def poll(self) -> dict | None:
        """Returns dict with model_bytes + step if available, else None."""
        try:
            _, payload = self.sock.recv_multipart(zmq.NOBLOCK)
            return pickle.loads(payload)
        except zmq.Again:
            return None
