# Distributed Training Plan — Pi5 Car + Server GPU

Real-world Dreamer training: Pi5 runs TFLite inference and collects experience,
RTX 4060 server runs world-model + actor + critic training. Connected over WiFi.
PS4 controller replaces keyboard human override.

**Constraint**: Pi5 only runs TFLite models. PyTorch (.pth) and H5 are too heavy.
ONNX ruled out. Conversion path: PyTorch → TFLite via `ai-edge-torch` (Google,
converts nn.Module directly — no ONNX intermediate).

---

## Architecture

```
┌──────────────────────────────────────────┐
│              Pi5 (Car)                   │
│                                          │
│  PiCamera2 → preprocess (crop/resize)    │
│       ↓                                  │
│  inference.tflite                        │
│  (encoder + GRU step + actor, fused)     │
│  inputs:  obs, prev_belief, prev_state,  │
│           prev_action                    │
│  outputs: action, new_belief, new_state  │
│       ↓                                  │
│  Steering PWM → DonkeyCar actuator       │
│                                          │
│  PS4 Controller                          │
│    ○ Circle  → STOP  (penalty -1)        │
│    × Cross   → RESET (clean lap,  0)     │
│    △ Triangle→ QUIT  (end session)       │
│    □ Square  → PAUSE (hold zero throttle)│
│                                          │
│  Episode buffer → ZMQ PUSH ─────────────┼──▶ Server
│  ZMQ SUB ◀───────────────────────────────┼─── Server
│  (receive inference.tflite bytes)        │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│         Server (RTX 4060)                │
│                                          │
│  ZMQ PULL → append to replay buffer      │
│  Dreamer.update_parameters() [PyTorch]   │
│    world model + actor + critic          │
│  Every N episodes:                       │
│    export PyTorch weights                │
│    ai_edge_torch → inference.tflite      │
│    ZMQ PUB → send .tflite to Pi5 ───────┼──▶ Pi5
│                                          │
│  W&B / CSV logging                       │
│  Checkpoint .pth saves                   │
└──────────────────────────────────────────┘
```

---

## TFLite Inference Model

The key insight: Pi5 only needs to run the **inference graph** — not the full
Dreamer training loop. This is 3 small models fused into one stateful TFLite:

```
obs [C,64,64] ──▶ Encoder CNN ──▶ embedding [1024]
                                       │
prev_belief [200] ──────────────────── │
prev_state  [30]  ──▶ GRU + posterior ─┤──▶ new_belief [200]
prev_action [2]   ──────────────────── │    new_state  [30]
                                       │
                               new_belief + new_state
                                       │
                                  Actor MLP
                                       │
                                  action [2]
```

### `dreamer/export_tflite.py`  (runs on server)

```python
import torch, ai_edge_torch
from dreamer.models import Encoder, TransitionModel, ActorModel

class InferenceGraph(torch.nn.Module):
    """
    Fused single-step inference: obs → action, next belief/state.
    Stateless — caller passes in prev_belief and prev_state each step.
    This design makes the TFLite model simple (no internal state to manage).
    """
    def __init__(self, encoder, transition, actor):
        super().__init__()
        self.encoder    = encoder
        self.transition = transition
        self.actor      = actor

    def forward(self, obs, prev_belief, prev_state, prev_action):
        # obs: [1, C, 64, 64]
        embedding = self.encoder(obs)                         # [1, 1024]
        # Single GRU step (posterior — uses real observation)
        hidden = torch.tanh(self.transition.fc_embed_state_action(
            torch.cat([prev_state, prev_action], dim=1)
        ))
        new_belief = self.transition.rnn_norm(
            self.transition.rnn(hidden, prev_belief)
        )
        post_hidden = torch.tanh(self.transition.fc_embed_belief_posterior(
            torch.cat([new_belief, embedding], dim=1)
        ))
        post_out = self.transition.fc_state_posterior(post_hidden)
        post_mean, post_std_raw = torch.chunk(post_out, 2, dim=1)
        post_std = torch.nn.functional.softplus(post_std_raw) + 0.1
        new_state = post_mean + post_std * torch.randn_like(post_mean)
        # Actor (deterministic = True at inference time)
        action, _ = self.actor(new_belief, new_state,
                               deterministic=True, with_logprob=False)
        return action, new_belief, new_state


def export_tflite(agent, args, output_path='inference.tflite'):
    graph = InferenceGraph(
        agent.encoder, agent.transition_model, agent.actor_model
    ).eval().cpu()

    C = args.observation_size[0]
    sample_inputs = (
        torch.zeros(1, C, 64, 64),          # obs
        torch.zeros(1, args.belief_size),   # prev_belief
        torch.zeros(1, args.state_size),    # prev_state
        torch.zeros(1, args.action_size),   # prev_action
    )

    edge_model = ai_edge_torch.convert(graph, sample_inputs)
    edge_model.export(output_path)
    print(f'Exported TFLite model → {output_path}')
    return output_path
```

### Pi5 TFLite Inference

```python
import numpy as np
import tflite_runtime.interpreter as tflite

class TFLiteInference:
    def __init__(self, model_path, belief_size=200, state_size=30, action_size=2):
        self.interp = tflite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp  = {d['name']: d for d in self.interp.get_input_details()}
        self.out  = {d['name']: d for d in self.interp.get_output_details()}
        self.belief = np.zeros((1, belief_size), dtype=np.float32)
        self.state  = np.zeros((1, state_size),  dtype=np.float32)
        self.action = np.zeros((1, action_size), dtype=np.float32)

    def step(self, obs: np.ndarray):
        """obs: [C, 64, 64] float32 [-0.5, 0.5] → action [2] float32"""
        self.interp.set_tensor(self.inp['obs']['index'],
                               obs[np.newaxis])           # add batch dim
        self.interp.set_tensor(self.inp['prev_belief']['index'], self.belief)
        self.interp.set_tensor(self.inp['prev_state']['index'],  self.state)
        self.interp.set_tensor(self.inp['prev_action']['index'], self.action)
        self.interp.invoke()
        self.action = self.interp.get_tensor(self.out['action']['index'])
        self.belief = self.interp.get_tensor(self.out['new_belief']['index'])
        self.state  = self.interp.get_tensor(self.out['new_state']['index'])
        return self.action[0]  # [2] — [steering, throttle]

    def reset_state(self):
        self.belief[:] = 0.0
        self.state[:]  = 0.0
        self.action[:]  = 0.0

    def reload(self, model_bytes: bytes):
        """Hot-reload model from new bytes without restarting."""
        with open('/tmp/inference_new.tflite', 'wb') as f:
            f.write(model_bytes)
        self.interp = tflite.Interpreter(model_path='/tmp/inference_new.tflite')
        self.interp.allocate_tensors()
        self.inp = {d['name']: d for d in self.interp.get_input_details()}
        self.out = {d['name']: d for d in self.interp.get_output_details()}
```

---

## Data Flow

### Experience (Pi5 → Server)
Sent at end of each episode as a single ZMQ message:

```python
{
  'observations': np.array,  # [T, C, 64, 64] float32 [-0.5, 0.5]
  'actions':      np.array,  # [T, 2]         float32
  'rewards':      np.array,  # [T]            float32
  'dones':        np.array,  # [T]            bool
  'episode_num':  int,
  'steps':        int,
}
```

Estimated size per episode (500 steps, RGB):
- obs: 500 × 3 × 64 × 64 × 4 bytes ≈ 24 MB
- rest: negligible
- Total: ~24 MB/episode — ~2 seconds on typical home WiFi (100 Mbps)

For `--grayscale`: ~8 MB/episode.

### Model (Server → Pi5)
Sent after every `weight_push_interval` episodes:

```python
{
  'model_bytes': bytes,  # raw .tflite file contents (~2–5 MB)
  'step':        int,
}
```

Pi5 hot-reloads interpreter without restarting the control loop.

---

## Network Setup

```
Server IP: static local IP e.g. 192.168.1.100  (set in router or /etc/hosts)
Pi5 IP:    DHCP or static e.g. 192.168.1.101

Ports:
  5555 — experience  (Pi5 PUSH → Server PULL)
  5556 — model       (Server PUB → Pi5 SUB)
```

Assign static IPs via router DHCP reservation using MAC addresses.
Verify connectivity: `ping 192.168.1.100` from Pi5 before starting.

---

## ZMQ Communication Layer — `dreamer/comms.py`

Shared by Pi5 and server. Pi5 does NOT need PyTorch — only `pyzmq`,
`tflite_runtime`, `inputs`, `picamera2`, and `numpy`.

```python
import zmq, pickle, numpy as np

EXPERIENCE_PORT = 5555
MODEL_PORT      = 5556

class ExperienceSender:
    """Pi5 — pushes episode buffer to server after each episode."""
    def __init__(self, server_ip):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.connect(f"tcp://{server_ip}:{EXPERIENCE_PORT}")

    def send(self, obs, actions, rewards, dones, meta):
        self.sock.send(pickle.dumps(
            {'obs': obs, 'actions': actions, 'rewards': rewards,
             'dones': dones, **meta}
        ))

class ExperienceReceiver:
    """Server — receives episode data from Pi5."""
    def __init__(self):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PULL)
        self.sock.bind(f"tcp://*:{EXPERIENCE_PORT}")

    def recv(self):
        return pickle.loads(self.sock.recv())

class ModelPublisher:
    """Server — broadcasts updated .tflite bytes after export."""
    def __init__(self):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{MODEL_PORT}")

    def publish(self, tflite_path: str, step: int):
        with open(tflite_path, 'rb') as f:
            model_bytes = f.read()
        payload = pickle.dumps({'model_bytes': model_bytes, 'step': step})
        self.sock.send_multipart([b'model', payload])
        print(f'[Server] Published model update — step {step}, '
              f'size {len(model_bytes)/1024:.0f} KB')

class ModelSubscriber:
    """Pi5 — polls for updated model bytes from server."""
    def __init__(self, server_ip):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://{server_ip}:{MODEL_PORT}")
        self.sock.setsockopt(zmq.SUBSCRIBE, b'model')

    def poll(self):
        """Non-blocking. Returns dict with model_bytes if available, else None."""
        try:
            _, payload = self.sock.recv_multipart(zmq.NOBLOCK)
            return pickle.loads(payload)
        except zmq.Again:
            return None
```

---

## PS4 Controller — `dreamer/envs/ps4_override.py`

`inputs` library uses Linux evdev — no root, no udev rules on Pi5.

```bash
# Pi5
pip install inputs
# Pair controller: bluetoothctl → scan on → pair <MAC> → connect <MAC>
```

Button mapping:

| Button     | Event  | Reward | Terminates |
|------------|--------|--------|------------|
| ○ Circle   | STOP   | -1.0   | yes        |
| × Cross    | RESET  | 0.0    | yes        |
| △ Triangle | QUIT   | 0.0    | yes (session end) |
| □ Square   | PAUSE  | —      | no — holds zero throttle |

Same `consume_event()` / `should_quit` interface as `HumanOverride` — no changes
needed in the episode collection loop.

```python
import threading

class PS4Override:
    STOP  = 'stop'
    RESET = 'reset'
    QUIT  = 'quit'

    def __init__(self):
        self._event  = None
        self._paused = False
        self._lock   = threading.Lock()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        print('[PS4Override] Listening — ○=STOP  ×=RESET  △=QUIT  □=PAUSE')

    def _poll_loop(self):
        from inputs import get_gamepad
        BTN_MAP = {
            ('BTN_EAST',  1): self.STOP,   # ○ Circle
            ('BTN_SOUTH', 1): self.RESET,  # × Cross
            ('BTN_NORTH', 1): self.QUIT,   # △ Triangle
        }
        while True:
            for ev in get_gamepad():
                key = (ev.code, ev.state)
                if key == ('BTN_WEST', 1):     # □ Square — toggle pause
                    with self._lock:
                        self._paused = not self._paused
                        print(f'\r[PS4Override] {"PAUSED" if self._paused else "RESUMED"}   ')
                elif key in BTN_MAP:
                    with self._lock:
                        self._event = BTN_MAP[key]
                        print(f'\r[PS4Override] {BTN_MAP[key].upper()}   ')

    def consume_event(self):
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
```

---

## Pi5 Inference Script — `car_inference.py`

No PyTorch on Pi5. Only: `tflite_runtime`, `picamera2`, `pyzmq`, `inputs`, `numpy`, `opencv-python`.

### Camera Setup

```python
from picamera2 import Picamera2
import cv2, numpy as np

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"size": (160, 120), "format": "RGB888"}
))
cam.start()

def capture_obs(grayscale=False):
    frame = cam.capture_array()          # [120, 160, 3] uint8
    frame = frame[40:, :, :]             # crop top 40 rows → [80, 160, 3]
    frame = cv2.resize(frame, (64, 64))  # → [64, 64, 3]
    if grayscale:
        frame = np.dot(frame, [0.299, 0.587, 0.114]).astype(np.float32)
        obs = frame / 255.0 - 0.5        # [64, 64] → expand later
        return obs[np.newaxis]           # [1, 64, 64]
    else:
        obs = frame.astype(np.float32) / 255.0 - 0.5
        return obs.transpose(2, 0, 1)   # [3, 64, 64]
```

### Actuator Setup

```python
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

steering_controller = PCA9685(channel=1, busnum=1)
throttle_controller = PCA9685(channel=0, busnum=1)
steering = PWMSteering(controller=steering_controller)
throttle = PWMThrottle(controller=throttle_controller)

def send_action(action, fix_speed=True, throttle_base=0.3):
    steering.run(float(action[0]))
    throttle.run(throttle_base if fix_speed else float(action[1]))

def send_zero():
    steering.run(0.0)
    throttle.run(0.0)
```

### Control Loop (10 Hz)

```python
import time
from dreamer.comms import ExperienceSender, ModelSubscriber
from dreamer.envs.ps4_override import PS4Override

SERVER_IP = '192.168.1.100'
STEP_DT   = 0.1   # 10 Hz

ps4      = PS4Override()
sender   = ExperienceSender(SERVER_IP)
model_sub = ModelSubscriber(SERVER_IP)
model    = TFLiteInference('inference.tflite')  # initial model from sim

obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
episode_num = 0

while not ps4.should_quit:
    t0 = time.monotonic()

    obs = capture_obs(grayscale=False)   # match --grayscale flag used on server

    if ps4.is_paused:
        send_zero()
        time.sleep(max(0, STEP_DT - (time.monotonic() - t0)))
        continue

    action = model.step(obs)             # TFLite inference
    send_action(action)

    ev     = ps4.consume_event()
    reward = -1.0 if ev == 'stop' else (0.0 if ev in ('reset', 'quit') else 1.0)
    done   = ev in ('stop', 'reset', 'quit')

    obs_buf.append(obs)
    act_buf.append(action.copy())
    rew_buf.append(reward)
    done_buf.append(done)

    if done:
        send_zero()
        sender.send(
            obs=np.stack(obs_buf),
            actions=np.stack(act_buf),
            rewards=np.array(rew_buf, dtype=np.float32),
            dones=np.array(done_buf, dtype=bool),
            meta={'episode_num': episode_num, 'steps': len(rew_buf)},
        )
        print(f'[Car] Episode {episode_num} sent — '
              f'{len(rew_buf)} steps, reward {sum(rew_buf):.1f}')
        obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
        model.reset_state()
        episode_num += 1

    # Non-blocking model update check
    update = model_sub.poll()
    if update:
        model.reload(update['model_bytes'])
        print(f'[Car] Model updated — server step {update["step"]}')

    elapsed = time.monotonic() - t0
    time.sleep(max(0, STEP_DT - elapsed))

send_zero()
cam.stop()
```

---

## Server Training Script — `server_trainer.py`

Fully automated. Launch once, runs until you kill it.

```bash
python server_trainer.py \
    --server_ip 0.0.0.0 \
    --episodes 500 \
    --weight_push_interval 2 \
    --min_buffer_steps 1000 \
    --augment
```

```python
from dreamer.comms import ExperienceReceiver, ModelPublisher
from dreamer.export_tflite import export_tflite
from dreamer.agent import Dreamer

receiver  = ExperienceReceiver()
publisher = ModelPublisher()
agent     = Dreamer(args)

# Optional: bootstrap from sim checkpoint
if args.models:
    agent.load_checkpoint(args.models)
    print(f'[Server] Loaded sim checkpoint: {args.models}')

episode_count = 0

while episode_count < args.episodes:
    print(f'[Server] Waiting for episode {episode_count + 1}...')
    ep = receiver.recv()

    # Append to replay buffer
    T = len(ep['rewards'])
    for t in range(T):
        agent.D.append(ep['obs'][t], ep['actions'][t],
                       ep['rewards'][t], ep['dones'][t])

    episode_count += 1
    print(f'[Server] Episode {ep["episode_num"]} received — '
          f'{T} steps, reward {ep["rewards"].sum():.1f}, '
          f'buffer {agent.D.steps} steps')

    # Train if buffer warm
    if agent.D.steps >= args.min_buffer_steps:
        loss_info = agent.update_parameters(args.collect_interval)
        log_to_wandb(loss_info, episode_count)

    # Export + push model every N episodes
    if episode_count % args.weight_push_interval == 0:
        tflite_path = export_tflite(agent, args,
                                    output_path=f'results/inference_{episode_count}.tflite')
        publisher.publish(tflite_path, step=agent.D.steps)

    # Checkpoint PyTorch weights
    if episode_count % args.checkpoint_interval == 0:
        agent.save_checkpoint(f'results/checkpoint_{episode_count}.pth')
```

---

## Implementation Phases

### Phase 0 — TFLite Export Pipeline (server only, no car needed)
- [ ] Install `ai-edge-torch` on server: `pip install ai-edge-torch`
- [ ] Write `dreamer/export_tflite.py` — `InferenceGraph` + `export_tflite()`
- [ ] Export current sim-trained checkpoint → `inference.tflite`
- [ ] Verify on server: load with `tflite_runtime`, run fake input, check output shapes
- [ ] Copy `.tflite` to Pi5, test load with `tflite_runtime` on Pi5

### Phase 1 — Communication Layer
- [ ] Write `dreamer/comms.py`
- [ ] Loopback test on single machine: fake episode PUSH → PULL, fake model PUB → SUB
- [ ] Cross-device test: Pi5 sends numpy arrays, server receives and prints shape
- [ ] Measure WiFi throughput for 24 MB episode payload (target < 5 sec)

### Phase 2 — PS4 Controller on Pi5
- [ ] `pip install inputs` on Pi5
- [ ] Pair PS4 via Bluetooth: `bluetoothctl`
- [ ] Write `dreamer/envs/ps4_override.py`
- [ ] Test all 4 buttons, verify `consume_event()` returns correct values
- [ ] Test PAUSE toggle: car should hold zero throttle

### Phase 3 — Pi5 Inference Node
- [ ] Write `car_inference.py`
- [ ] Test PiCamera2 capture + preprocessing pipeline (save sample frames)
- [ ] Profile TFLite inference latency on Pi5: target < 50 ms/step
- [ ] Wire up DonkeyCar actuators (steering + throttle PWM)
- [ ] Static test: run inference on dummy input, verify actuators move
- [ ] Dynamic test: drive with sim-bootstrapped `.tflite`, observe steering

### Phase 4 — Server Training Node
- [ ] Write `server_trainer.py`
- [ ] Offline test: feed 5 recorded episodes from Pi5, verify training runs
- [ ] Verify export → publish → Pi5 reload round-trip
- [ ] Check W&B logs appear correctly

### Phase 5 — Integration
- [ ] End-to-end dry run: Pi5 sends fake episodes, server trains, sends `.tflite` back
- [ ] First real seed collection: 10 episodes, PS4 operator, no training yet
- [ ] First live training run with model pushback
- [ ] Verify car behaviour improves across episodes

### Phase 6 — Automation & Monitoring
- [ ] Auto-reconnect on WiFi drop (ZMQ handles natively)
- [ ] W&B dashboard: reward curve, losses, model push events
- [ ] Pi5 audio beep / LED flash on model update received
- [ ] Graceful QUIT: drain buffer → send → server checkpoint → exit

---

## Dependencies

### Pi5 (minimal — no PyTorch)
```bash
pip install tflite-runtime pyzmq inputs picamera2 numpy opencv-python
# donkeycar already installed
```

### Server
```bash
pip install torch torchvision ai-edge-torch pyzmq wandb tqdm
```

---

## Key Hyperparameter Adjustments for Real World

| Parameter | Sim default | Real-world suggestion | Reason |
|---|---|---|---|
| `seed_episodes` | 5 | 10–20 | Real data harder to collect |
| `collect_interval` | 100 | 200–400 | Server has time while Pi5 drives |
| `weight_push_interval` | — | every 2 episodes | Fresh model matters more in real world |
| `min_buffer_steps` | — | 1000 | Don't train on too little data |
| `expl_amount` | 0.15 | 0.0 | TFLite runs deterministic; exploration via environment randomness |
| `smooth_weight` | 0.05 | 0.1 | Protect real servo from jitter |
| `augment` | optional | **always on** | Sim-to-real gap is the whole problem |
| `max_episode_length` | 1000 | 500 | Shorter = faster real-world iteration |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `ai-edge-torch` can't export GRUCell | Use `torch.jit.trace` + manual GRU unroll in InferenceGraph |
| WiFi dropout mid-episode | ZMQ PUSH queues locally; auto-reconnects |
| Pi5 inference > 50 ms | Reduce image to 32×32 for TFLite only; train server at 64×64 |
| Car crashes before STOP pressed | Add bumper switch wired to GPIO → auto STOP signal |
| Stale model makes car worse | Keep last 3 `.tflite` snapshots; manual rollback if needed |
| PS4 Bluetooth drops | Use USB Bluetooth adapter (< $10) for reliable < 5ms latency |
| Server export blocks training | Run `export_tflite` in a background thread |

---

## Open Questions

1. **Throttle**: Fix speed (`--fix_speed`) recommended for real world until model is stable.
2. **Bootstrapping**: Use current sim-trained `.pth` checkpoint to generate first `.tflite`.
   Strong recommendation — avoids random flailing on real hardware during seed episodes.
3. **Episode reset**: Car must be manually placed at start. Consider painted start marker
   as visual cue for the operator.
4. **Reward signal**: Pure PS4 operator (survival) to start. Vision-based lane reward
   can be added later once model is stable — same `_visual_cte()` function already exists.
5. **ai-edge-torch GRU support**: Needs validation. If GRU export fails, replace with
   a manually unrolled single GRU cell using only basic matmul + tanh ops (all TFLite-native).
