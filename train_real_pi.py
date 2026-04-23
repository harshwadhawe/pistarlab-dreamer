"""
Pi5 inference + experience collection loop for real-world DonkeyCar.

All settings are in config.toml [pi] section.
Edit server_ip in config.toml before each session.

Usage:
  python train_real_pi.py
"""

import os
import time
import warnings

import numpy as np
import tflite_runtime.interpreter as tflite

warnings.filterwarnings('ignore', category=DeprecationWarning)

from dreamer.config import load_config
from dreamer.utils.obs import preprocess_frame
from dreamer.comms import make_comms
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
        self._inp = self._map_by_shape(self.interp.get_input_details())
        self._out = self._map_by_shape(self.interp.get_output_details())

    def _map_by_shape(self, details):
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
        return self.action[0]

    def reload(self, model_bytes: bytes):
        tmp = 'models/inference_active.tflite'
        os.makedirs('models', exist_ok=True)
        with open(tmp, 'wb') as f:
            f.write(model_bytes)
        self._load(tmp)
        print('[Pi] New model loaded and active.')


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
            model_path = args.model or (
                f'models/inference_{"grayscale" if args.grayscale else "rgb"}.tflite'
            )
            print(f'[INFO] Loading TFLite model: {model_path}')
            self.model = DreamerTFLite(
                model_path=model_path,
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
            from dreamer.envs.controller import EpisodeController
            self.ps4 = EpisodeController.from_gamepad()

            print('[OK] All systems nominal.\n')

        except Exception as e:
            print(f'\n[FATAL] Init failed: {e}')
            self.shutdown()
            raise SystemExit(1)

    def preprocess(self, frame):
        return preprocess_frame(frame, self.args.channels)

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
        print('  R1         → START   next episode')
        print('  ○ Circle   → STOP    (off-track, reward -1)')
        print('  × Cross    → RESET   (clean lap,  reward +1)')
        print('  □ Square   → DISCARD (erase episode, retry)')
        print('  Ctrl+C to stop session.')
        print('='*52 + '\n')

        episode_num = 1

        while True:
            print(f'[Car] Episode {episode_num} — running...')

            discarded = True
            while discarded:
                obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
                self.model.reset_state()
                discarded = False

                while True:
                    t0 = time.time()

                    frame  = self.camera.run()
                    obs    = self.preprocess(frame)
                    action = self.model.step(obs)

                    if episode_num <= self.args.seed_episodes:
                        if not hasattr(self, '_seed_steer') or \
                                time.time() - self._seed_steer_t >= 0.5:
                            self._seed_steer   = float(np.random.uniform(-1.0, 1.0))
                            self._seed_steer_t = time.time()
                        action = action.copy()
                        action[0] = self._seed_steer

                    s, t = self.send_action(float(action[0]), float(action[1]))

                    ev = self.ps4.consume_event()
                    reward, done, discard = self.ps4.event_to_outcome(ev)
                    if discard:
                        self.send_zero()
                        print('\n[Car] DISCARD — episode erased, retrying...')
                        if self.args.server_ip:
                            self.sender.send_discard(episode_num)
                        discarded = True
                        break

                    if len(rew_buf) >= self.args.max_episode_steps:
                        reward, done = 1.0, True

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

            if self.args.server_ip and episode_num > self.args.seed_episodes:
                print('[Car] Waiting for server to finish training + export...')
                wait_start = time.time()
                while True:
                    self.send_zero()
                    update = self.model_sub.poll()
                    if update:
                        kb = len(update['model_bytes']) / 1024
                        print(f'[Car] Receiving weights ({kb:.0f} KB)...', end=' ', flush=True)
                        self.model.reload(update['model_bytes'])
                        print(f'done. (server step {update["step"]})')
                        break
                    elapsed = time.time() - wait_start
                    if int(elapsed) % 30 == 0 and elapsed > 5:
                        print(f'[Car] Still waiting for server... {elapsed:.0f}s elapsed', end='\r')
                    if elapsed > 600:
                        print(f'\n[Car] WARNING: no model from server after {elapsed:.0f}s — '
                              f'check server is running and rsync from {self.args.server_ip} is reachable.')
                        wait_start = time.time()
                    if self.ps4.should_quit:
                        break
                    time.sleep(0.05)

            self.ps4.flush()
            print('[Car] Press R1 to start next episode...')
            while True:
                self.send_zero()
                ev = self.ps4.consume_event()
                if ev == self.ps4.START:
                    break
                time.sleep(0.05)

        self.ps4.close()
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
    args = load_config('pi')
    car = PhysicalDreamerCar(args)
    car.run()
