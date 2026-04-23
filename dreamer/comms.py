"""
Communication layer — server-side rsync replaces ZMQ + HTTP.

Episode transport (Pi → Server):
  Pi writes compressed npz + ready sentinel to PI_OUTBOX on Pi filesystem.
  Server rsync-pulls from car:PI_OUTBOX into a local inbox directory.
  obs saved as uint8 to cut 96 MB RGB episode to ~24 MB before compression.

Model transport (Server → Pi):
  Server rsync-pushes latest.tflite then step.txt to car:PI_INBOX.
  Pi polls PI_INBOX/step.txt — only reads model after step advances,
  ensuring tflite is fully synced before Pi loads it.

SSH alias for Pi is configured in config.toml [real] pi_alias.
"""

import os
import subprocess
import time

import numpy as np

PI_OUTBOX      = '/tmp/dreamer/outbox'
PI_INBOX       = '/tmp/dreamer/inbox'
MODEL_HTTP_PORT = 5557   # used only by push_init_model.py / pull_init_model.py


# ---------------------------------------------------------------------------
# Null-object implementations — standalone mode (no server)
# ---------------------------------------------------------------------------

class NoopSender:
    def send(self, **_): pass
    def send_discard(self, episode_num: int): pass


class NoopSubscriber:
    def poll(self): return None


def make_comms(server_ip: str):
    """Return (sender, subscriber) — rsync-local pair or no-ops if server_ip empty."""
    if not server_ip:
        return NoopSender(), NoopSubscriber()
    return RsyncExperienceSender(), RsyncModelWatcher()


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
        print(f'[Comms] RsyncExperienceSender outbox → {PI_OUTBOX}')

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
        os.makedirs(PI_OUTBOX, exist_ok=True)
        sentinel = os.path.join(PI_OUTBOX, f'ep_{episode_num:04d}.discard')
        open(sentinel, 'w').close()
        print(f'[Pi] Discard signal → {sentinel}')


class RsyncModelWatcher:
    """Pi — polls PI_INBOX for updated model bytes pushed by server."""

    def __init__(self):
        os.makedirs(PI_INBOX, exist_ok=True)
        self._last_step = -1
        print(f'[Comms] RsyncModelWatcher inbox → {PI_INBOX}')

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
        print(f'[Pi ← Server] New model received — step={step}  '
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
        print(f'[Comms] RsyncExperienceReceiver: rsync {pi_alias}:{PI_OUTBOX}/ → {local_inbox}/')

    def recv(self) -> dict:
        """Block until a new episode or discard notification arrives from Pi."""
        while True:
            subprocess.run([
                'rsync', '-az', '--timeout=10',
                f'{self.pi_alias}:{PI_OUTBOX}/',
                f'{self.local_inbox}/',
            ], capture_output=True)

            # Discard sentinels take priority
            for fname in sorted(os.listdir(self.local_inbox)):
                if fname.endswith('.discard') and fname not in self._seen:
                    self._seen.add(fname)
                    ep_num = int(fname.split('_')[1].split('.')[0])
                    print(f'[Comms] Episode {ep_num} discarded by operator.')
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
    """Server — pushes TFLite to Pi via rsync. tflite synced before step.txt
    so Pi never loads a partially-transferred model."""

    def __init__(self, pi_alias: str, progress: bool = True):
        self.pi_alias = pi_alias
        self.progress = progress
        print(f'[Comms] RsyncModelPublisher → {pi_alias}:{PI_INBOX}/')

    def publish(self, tflite_path: str, step: int):
        # Push model first — Pi won't load it until step.txt advances
        kb = os.path.getsize(tflite_path) // 1024
        print(f'[Server → Pi] Pushing model ({kb} KB) — step {step}...')
        cmd = ['rsync', '-az', '--timeout=60']
        if self.progress:
            cmd.append('--progress')
        r1 = subprocess.run(cmd + [tflite_path, f'{self.pi_alias}:{PI_INBOX}/latest.tflite'])
        if r1.returncode != 0:
            print(f'[Comms] rsync model FAILED (code {r1.returncode})')
            return

        step_tmp = '/tmp/dreamer_push_step.txt'
        with open(step_tmp, 'w') as f:
            f.write(str(step))
        r2 = subprocess.run([
            'rsync', '-az', '--timeout=10',
            step_tmp,
            f'{self.pi_alias}:{PI_INBOX}/step.txt',
        ], capture_output=True)
        if r2.returncode != 0:
            print(f'[Comms] rsync step.txt FAILED (code {r2.returncode})')
        else:
            print(f'[Server → Pi] Model live on Pi.')
