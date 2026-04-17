"""
Run the Dreamer inference loop on the physical DonkeyCar using a fused TFLite model.

The TFLite model takes:
  obs [1,C,64,64], prev_belief [1,200], prev_state [1,30], prev_action [1,2]
and returns:
  action [1,2], new_belief [1,200], new_state [1,30]

Belief and state carry across steps and reset at each episode boundary.

With --server_ip set, the car also:
  - Sends each completed episode to the server trainer via ZMQ PUSH
  - Polls for updated TFLite weights and hot-reloads them mid-run

Without --server_ip, runs as a standalone inference loop (no comms).

Run on Pi5 (no PyTorch needed):
  pip install tflite-runtime numpy opencv-python pyzmq donkeycar

Usage:
  # Standalone inference (no server):
  python drive_physical_tflite.py --model inference.tflite

  # Connected to server trainer:
  python drive_physical_tflite.py --model inference.tflite --server_ip 192.168.1.100

  # RGB model:
  python drive_physical_tflite.py --model inference.tflite --server_ip 192.168.1.100 --channels 3
"""

import argparse
import os
import sys
import time

# Ensure repo root is on path when running as scripts/drive_physical_tflite.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import warnings
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

warnings.filterwarnings('ignore', category=DeprecationWarning)

from donkeycar.parts.actuator import PCA9685
from donkeycar.parts.camera import PiCamera

# --- Hardware config (match your physical wiring) ---
STEERING_CHANNEL  = 14
THROTTLE_CHANNEL  = 1
I2C_BUSNUM        = 1

STEERING_LEFT_PWM    = 470
STEERING_RIGHT_PWM   = 320
THROTTLE_FORWARD_PWM = 480
THROTTLE_STOPPED_PWM = 400
THROTTLE_REVERSE_PWM = 320

# --- Tuning ---
IMG_CROP_TOP   = 40       # rows to crop from top of 120-row frame → 80 rows remain
STEERING_GAIN  = 1.0
THROTTLE_BOOST = 1.0


def pwm_map(val, val_min, val_max, pwm_min, pwm_max):
    return int((val - val_min) * (pwm_max - pwm_min) / (val_max - val_min) + pwm_min)


# ---------------------------------------------------------------------------
# TFLite inference wrapper
# ---------------------------------------------------------------------------

class DreamerTFLite:
    def __init__(self, model_path, belief_size=200, state_size=30, action_size=2):
        self.belief_size = belief_size
        self.state_size  = state_size
        self.action_size = action_size
        self._load(model_path)
        self.reset_state()

    def _load(self, path):
        self.interp = tflite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        # litert_torch generates generic names (serving_default_args_N:0).
        # Map by shape instead — each tensor has a unique shape.
        self._inp = self._map_by_shape(self.interp.get_input_details())
        self._out = self._map_by_shape(self.interp.get_output_details())

    def _map_by_shape(self, details):
        """Return dict keyed by tuple(shape) → tensor index."""
        return {tuple(d['shape'].tolist()): d['index'] for d in details}

    def _idx_in(self, shape):
        return self._inp[tuple(shape)]

    def _idx_out(self, shape):
        return self._out[tuple(shape)]

    def reset_state(self):
        self.belief = np.zeros((1, self.belief_size), dtype=np.float32)
        self.state  = np.zeros((1, self.state_size),  dtype=np.float32)
        self.action = np.zeros((1, self.action_size), dtype=np.float32)

    def step(self, obs: np.ndarray):
        """obs: [C, 64, 64] float32 [-0.5, 0.5] → action [2]"""
        C = obs.shape[0]
        self.interp.set_tensor(self._idx_in([1, C, 64, 64]),          obs[np.newaxis])
        self.interp.set_tensor(self._idx_in([1, self.belief_size]),    self.belief)
        self.interp.set_tensor(self._idx_in([1, self.state_size]),     self.state)
        self.interp.set_tensor(self._idx_in([1, self.action_size]),    self.action)
        self.interp.invoke()
        self.action = self.interp.get_tensor(self._idx_out([1, self.action_size]))
        self.belief = self.interp.get_tensor(self._idx_out([1, self.belief_size]))
        self.state  = self.interp.get_tensor(self._idx_out([1, self.state_size]))
        return self.action[0]   # [2]: [steering, throttle]

    def reload(self, model_bytes: bytes):
        """Hot-reload weights without restarting the control loop."""
        tmp = '/tmp/inference_reload.tflite'
        with open(tmp, 'wb') as f:
            f.write(model_bytes)
        self._load(tmp)
        print('[Model] Hot-reloaded.')


# ---------------------------------------------------------------------------
# ZMQ comms (optional — only imported when --server_ip is set)
# ---------------------------------------------------------------------------

class NoopSender:
    def send(self, **_): pass


class NoopSubscriber:
    def poll(self): return None


def make_comms(server_ip):
    if not server_ip:
        return NoopSender(), NoopSubscriber()
    from dreamer.comms import ExperienceSender, ModelSubscriber
    return (
        ExperienceSender(server_ip),
        ModelSubscriber(server_ip),
    )


# ---------------------------------------------------------------------------
# Main car class
# ---------------------------------------------------------------------------

class PhysicalDreamerCar:
    def __init__(self, args):
        print('\n' + '='*52)
        print(f'  DREAMER TFLITE | channels={args.channels} boost={THROTTLE_BOOST}x')
        if args.server_ip:
            print(f'  Connected → server {args.server_ip}')
        else:
            print('  Standalone mode (no server)')
        print('='*52)
        self.args = args
        self.steering_ctrl = None
        self.throttle_ctrl = None
        self.camera        = None

        try:
            print('[INFO] Loading TFLite model...')
            self.model = DreamerTFLite(
                model_path=args.model,
                belief_size=args.belief_size,
                state_size=args.state_size,
                action_size=2,
            )

            print('[INFO] Binding I2C PWM controllers...')
            self.steering_ctrl = PCA9685(channel=STEERING_CHANNEL, busnum=I2C_BUSNUM)
            self.throttle_ctrl = PCA9685(channel=THROTTLE_CHANNEL,  busnum=I2C_BUSNUM)

            print('[INFO] Starting PiCamera (128x120)...')
            self.camera = PiCamera(image_w=128, image_h=120)
            time.sleep(2)

            self.sender, self.model_sub = make_comms(args.server_ip)

            print('[INFO] Connecting PS4 controller...')
            from dreamer.envs.ps4_override import PS4Override
            self.ps4 = PS4Override()

            print('[OK] All systems nominal.\n')

        except Exception as e:
            print(f'\n[FATAL] Init failed: {e}')
            self.shutdown()
            sys.exit(1)

    def preprocess(self, frame):
        """[120,128,3] uint8 → [C,64,64] float32 [-0.5,0.5]"""
        cropped = frame[IMG_CROP_TOP:IMG_CROP_TOP + 80, :, :]   # [80,128,3]
        resized = cv2.resize(cropped, (64, 64))                  # [64,64,3]
        if self.args.channels == 1:
            gray = np.dot(resized, [0.299, 0.587, 0.114]).astype(np.float32)
            return (gray / 255.0 - 0.5)[np.newaxis]             # [1,64,64]
        obs = resized.astype(np.float32) / 255.0 - 0.5
        return obs.transpose(2, 0, 1)                            # [3,64,64]

    def send_action(self, steering_val, throttle_val):
        s = float(np.clip(steering_val * STEERING_GAIN, -1.0, 1.0))
        t = float(np.clip(throttle_val * THROTTLE_BOOST,  0.0, 1.0))
        self.steering_ctrl.run(pwm_map(s, -1.0, 1.0, STEERING_LEFT_PWM,  STEERING_RIGHT_PWM))
        self.throttle_ctrl.run(pwm_map(t,  0.0, 1.0, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM))
        return s, t

    def send_zero(self):
        self.steering_ctrl.run(int((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2))
        self.throttle_ctrl.run(THROTTLE_STOPPED_PWM)

    def run(self):
        print('='*52)
        print('  AUTONOMOUS MODE')
        print('  R1         → START next episode')
        print('  ○ Circle   → STOP  (off-track, reward -1)')
        print('  × Cross    → RESET (clean lap,  reward  0)')
        print('  △ Triangle → QUIT  (end session)')
        print('  □ Square   → PAUSE (hold zero throttle)')
        print('='*52 + '\n')

        episode_num = 0

        while not self.ps4.should_quit:
            obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
            self.model.reset_state()

            print(f'[Car] Episode {episode_num} — running...')

            while True:
                t0 = time.time()

                # PAUSE — hold zero throttle until released
                if self.ps4.is_paused:
                    self.send_zero()
                    time.sleep(0.05)
                    continue

                frame  = self.camera.run()
                obs    = self.preprocess(frame)
                action = self.model.step(obs)

                # Seed episodes: override steering with uniform random [-1, 1]
                # held for ~0.5s so the car actually moves before switching.
                if episode_num < self.args.seed_episodes:
                    if not hasattr(self, '_seed_steer') or \
                            time.time() - self._seed_steer_t >= 0.5:
                        self._seed_steer   = float(np.random.uniform(-1.0, 1.0))
                        self._seed_steer_t = time.time()
                    action = action.copy()
                    action[0] = self._seed_steer

                s, t   = self.send_action(float(action[0]), float(action[1]))

                ev     = self.ps4.consume_event()
                if ev == self.ps4.STOP:
                    reward, done = -1.0, True
                elif ev == self.ps4.RESET:
                    reward, done =  0.0, True
                else:
                    reward, done =  1.0, False

                obs_buf.append(obs.copy())
                act_buf.append(action.copy())
                rew_buf.append(reward)
                done_buf.append(done)

                fps = 1.0 / max(time.time() - t0, 1e-6)
                print(
                    f'[Ep {episode_num}] FPS:{fps:4.1f} | '
                    f'Steer:{s:>6.3f} | Throt:{t:>5.3f} | '
                    f'Steps:{len(rew_buf):>4d}',
                    end='\r',
                )

                if done or self.ps4.should_quit:
                    break

            self.send_zero()

            if not obs_buf:
                break

            steps   = len(rew_buf)
            total_r = sum(rew_buf)
            print(f'\n[Car] Episode {episode_num} done — '
                  f'{steps} steps | reward {total_r:.1f}')

            if self.args.server_ip:
                self.sender.send(
                    obs=np.stack(obs_buf).astype(np.float32),
                    actions=np.stack(act_buf).astype(np.float32),
                    rewards=np.array(rew_buf, dtype=np.float32),
                    dones=np.array(done_buf, dtype=bool),
                    meta={'episode_num': episode_num, 'steps': steps},
                )
                print(f'[Car] Episode {episode_num} sent to server.')

            episode_num += 1

            if self.ps4.should_quit:
                break

            # Halt until server pushes a new model (blocks during training).
            # During seed episodes: skip halt so car runs freely like sim.
            # Keeps motors zeroed. Noop when running standalone.
            if self.args.server_ip and episode_num > self.args.seed_episodes:
                print('[Car] Waiting for server to finish training + export...')
                while True:
                    self.send_zero()
                    update = self.model_sub.poll()
                    if update:
                        kb = len(update['model_bytes']) / 1024
                        print(f'[Car] Receiving weights ({kb:.0f} KB)...', end=' ', flush=True)
                        self.model.reload(update['model_bytes'])
                        print(f'done. (server step {update["step"]})')
                        break
                    if self.ps4.should_quit:
                        break
                    time.sleep(0.05)

            # Wait for R1 before starting next episode
            print('[Car] Press R1 to start next episode | △ to quit...')
            while True:
                ev = self.ps4.consume_event()
                if ev == self.ps4.START:
                    break
                if ev == self.ps4.QUIT or self.ps4.should_quit:
                    break
                time.sleep(0.05)

        self.ps4.stop()
        self.shutdown()

    def shutdown(self):
        print('\n' + '='*52)
        print('  SAFE SHUTDOWN')
        print('='*52)
        if self.throttle_ctrl:
            try: self.throttle_ctrl.run(THROTTLE_STOPPED_PWM)
            except: pass
        if self.steering_ctrl:
            try: self.steering_ctrl.run(int((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2))
            except: pass
        if self.camera:
            try: self.camera.shutdown()
            except: pass
        print('[DONE] Motors off.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        default='./models/inference.tflite')
    parser.add_argument('--server_ip',    type=str,  default='',
                        help='Server IP/hostname. Omit for standalone mode.')
    parser.add_argument('--channels',      type=int,  default=1,   help='1=grayscale 3=RGB')
    parser.add_argument('--belief-size',   type=int,  default=200)
    parser.add_argument('--state-size',    type=int,  default=30)
    parser.add_argument('--seed-episodes', type=int,  default=5,
                        help='Run this many episodes freely before halting for model updates')
    args = parser.parse_args()

    car = PhysicalDreamerCar(args)
    car.run()
