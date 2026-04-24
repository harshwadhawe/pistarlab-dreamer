"""
Communication layer — rsync over SSH.

Episode transport (Pi → Server):
  Pi writes compressed npz + ready sentinel to PI_OUTBOX on Pi filesystem.
  Server rsync-pulls from car:PI_OUTBOX into a local inbox directory.
  obs saved as uint8 to cut 96 MB RGB episode to ~24 MB before compression.

Model transport (Server → Pi):
  Server stages latest.tflite + step.txt together and rsyncs in one SSH call.
  Pi polls PI_INBOX/step.txt — only reads model after step advances,
  ensuring tflite is fully synced before Pi loads it. latest.tflite sorts
  before step.txt alphabetically so rsync transfers the model file first.

SSH alias for Pi configured in config.toml [real] pi_alias.
"""

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

PI_OUTBOX = '/tmp/dreamer/outbox'
PI_INBOX  = '/tmp/dreamer/inbox'


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _obs_to_uint8(obs: np.ndarray) -> np.ndarray:
    return ((obs + 0.5) * 255).clip(0, 255).astype(np.uint8)


def _uint8_to_obs(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0 - 0.5


# ---------------------------------------------------------------------------
# Pi side — local file ops only (server handles all rsync)
# ---------------------------------------------------------------------------

class RsyncExperienceSender:
    """Pi — saves episode as compressed npz, signals server via sentinel file."""

    def __init__(self):
        os.makedirs(PI_OUTBOX, exist_ok=True)
        print(f'[Comms] Outbox → {PI_OUTBOX}')

    def send(self, obs: np.ndarray, actions: np.ndarray,
             rewards: np.ndarray, dones: np.ndarray, meta: dict):
        ep       = meta.get('episode_num', 0)
        tmp_path = os.path.join(PI_OUTBOX, f'.ep_{ep:04d}_tmp.npz')
        npz_path = os.path.join(PI_OUTBOX, f'ep_{ep:04d}.npz')
        sentinel = os.path.join(PI_OUTBOX, f'ep_{ep:04d}.ready')
        np.savez_compressed(tmp_path,
                            obs=_obs_to_uint8(obs),
                            actions=actions,
                            rewards=rewards,
                            dones=dones)
        os.replace(tmp_path, npz_path)   # atomic — server never reads partial file
        open(sentinel, 'w').close()
        kb    = os.path.getsize(npz_path) // 1024
        steps = len(rewards)
        print(f'[Pi → Server] ep={ep:04d}  steps={steps}  obs={obs.shape}  '
              f'file={os.path.basename(npz_path)}  ({kb} KB)')

    def send_discard(self, episode_num: int):
        sentinel = os.path.join(PI_OUTBOX, f'ep_{episode_num:04d}.discard')
        open(sentinel, 'w').close()
        print(f'[Pi] Discard → {sentinel}')


class RsyncModelWatcher:
    """Pi — polls PI_INBOX for updated model pushed by server."""

    def __init__(self):
        os.makedirs(PI_INBOX, exist_ok=True)
        self._last_step = -1
        print(f'[Comms] Inbox → {PI_INBOX}')

    def poll(self) -> dict | None:
        step_file   = os.path.join(PI_INBOX, 'step.txt')
        tflite_path = os.path.join(PI_INBOX, 'latest.tflite')
        if not os.path.exists(step_file) or not os.path.exists(tflite_path):
            return None
        try:
            step = int(open(step_file).read().strip())
        except Exception:
            return None
        if step <= self._last_step:
            return None
        self._last_step = step
        with open(tflite_path, 'rb') as f:
            model_bytes = f.read()
        kb = len(model_bytes) // 1024
        print(f'[Pi ← Server] New model — step={step}  '
              f'file={os.path.basename(tflite_path)}  ({kb} KB)')
        return {'model_bytes': model_bytes, 'step': step}


# ---------------------------------------------------------------------------
# Server side — rsync from/to Pi (SSH alias in config pi_alias)
# ---------------------------------------------------------------------------

class RsyncExperienceReceiver:
    """Server — polls Pi for new episodes via rsync. Blocks until one arrives."""

    def __init__(self, pi_alias: str, local_inbox: str):
        self.pi_alias    = pi_alias
        self.local_inbox = local_inbox
        os.makedirs(local_inbox, exist_ok=True)
        self._seen: set[str] = set()
        print(f'[Comms] Receiver: rsync {pi_alias}:{PI_OUTBOX}/ → {local_inbox}/')

    def recv(self) -> dict:
        """Block until a new episode or discard notification arrives from Pi."""
        while True:
            subprocess.run([
                'rsync', '-az', '--timeout=10',
                f'{self.pi_alias}:{PI_OUTBOX}/',
                f'{self.local_inbox}/',
            ], capture_output=True)

            for fname in sorted(os.listdir(self.local_inbox)):
                if fname.endswith('.discard') and fname not in self._seen:
                    self._seen.add(fname)
                    ep_num = int(fname.split('_')[1].split('.')[0])
                    print(f'[Server] Episode {ep_num} discarded by operator.')
                    return {'discarded': True, 'episode_num': ep_num}

            for fname in sorted(os.listdir(self.local_inbox)):
                if not fname.endswith('.ready') or fname in self._seen:
                    continue
                npz_name = fname.replace('.ready', '.npz')
                npz_path = os.path.join(self.local_inbox, npz_name)
                if not os.path.exists(npz_path):
                    continue
                self._seen.add(fname)
                self._seen.add(npz_name)
                ep_num = int(fname.split('_')[1].split('.')[0])
                data   = np.load(npz_path)
                kb     = os.path.getsize(npz_path) // 1024
                steps  = len(data['rewards'])
                print(f'[Server ← Pi] ep={ep_num:04d}  steps={steps}  '
                      f'obs={data["obs"].shape}  file={npz_name}  ({kb} KB)')
                return {
                    'obs':         _uint8_to_obs(data['obs']),
                    'actions':     data['actions'],
                    'rewards':     data['rewards'],
                    'dones':       data['dones'],
                    'discarded':   False,
                    'episode_num': ep_num,
                }

            time.sleep(1.0)


class RsyncModelPublisher:
    """Server — stages tflite + step.txt and pushes both in one rsync call."""

    def __init__(self, pi_alias: str, progress: bool = True):
        self.pi_alias = pi_alias
        self.progress = progress
        print(f'[Comms] Publisher → {pi_alias}:{PI_INBOX}/')

    def publish(self, tflite_path: str, step: int):
        kb = os.path.getsize(tflite_path) // 1024
        print(f'[Server → Pi] Pushing model ({kb} KB) — step {step}...')
        with tempfile.TemporaryDirectory() as staging:
            shutil.copy2(tflite_path, os.path.join(staging, 'latest.tflite'))
            with open(os.path.join(staging, 'step.txt'), 'w') as f:
                f.write(str(step))
            cmd = ['rsync', '-az', '--timeout=60']
            if self.progress:
                cmd.append('--progress')
            r = subprocess.run(cmd + [f'{staging}/', f'{self.pi_alias}:{PI_INBOX}/'])
        if r.returncode != 0:
            print(f'[Comms] rsync FAILED (code {r.returncode})')
        else:
            print(f'[Server → Pi] Model live on Pi — step {step}.')
