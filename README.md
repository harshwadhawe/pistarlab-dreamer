# DonkeyCar Dreamer

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Model-based RL for a DonkeyCar using the [Dreamer](https://arxiv.org/abs/1912.01603) algorithm. The agent learns a world model (RSSM) from raw camera images, then trains its policy entirely via latent imagination — no environment interaction during policy updates.

Supports two modes:
- **Sim** — Unity DonkeyCar simulator, full training loop on your machine
- **Real** — Pi5 car collects experience, server trains and pushes TFLite back

---

## Setup

**Requires Python 3.11** — `gym_donkeycar 1.3.1` depends on `asyncore` (removed in 3.12).

```bash
conda create -n donkeycar-dreamer python=3.11 -y
conda activate donkeycar-dreamer
pip install -r requirements.txt
pip install git+https://github.com/tawnkramer/gym-donkeycar.git
```

One conda env covers everything: sim training, real-world server, TFLite export, and ZMQ car comms.

---

## Common Options

These flags apply to both `train_sim.py` and `train_real.py`.

### Observation mode

| Flag | Effect |
|------|--------|
| *(default)* | 3-channel RGB, `64×64` |
| `--grayscale` | 1-channel grayscale, `64×64` — faster, lower memory |

```bash
# RGB (default)
python train_sim.py --episodes 1000

# Grayscale
python train_sim.py --episodes 1000 --grayscale
```

> Match `--grayscale` between sim training and real-world deployment — the car inference script must use the same channel count.

### Steering smoothness penalty

Penalises jitter using the **rolling standard deviation** of steering over the last N steps. High std = oscillating; smooth curves score lower penalty.

```
reward -= smooth_weight × std(steering_window)
```

| Flag | Default | Effect |
|------|---------|--------|
| `--smooth_weight` | `0.05` | Penalty scale. Set `0` to disable. |
| `--smooth_window` | `10` | Window size (steps). Longer = more memory of past jitter. |

```bash
# Default smoothing
python train_sim.py --episodes 1000

# Heavier penalty, longer memory
python train_sim.py --episodes 1000 --smooth_weight 0.1 --smooth_window 15

# Disable smoothing
python train_sim.py --episodes 1000 --smooth_weight 0
```

### Checkpoints & resume

```bash
# Resume training from checkpoint + saved replay buffer
python train_sim.py --episodes 1000 \
    --models results/donkey-generated-track-v0/42/models_500.pth \
    --experience-replay results/donkey-generated-track-v0/42/experience.pth

# Eval only (no environment interaction)
python train_sim.py --test \
    --models results/donkey-generated-track-v0/42/models_500.pth
```

### Key hyperparameters

| Flag | Default | Notes |
|------|---------|-------|
| `--episodes` | `30` | Use `1000+` for real training |
| `--seed-episodes` | `5` | Random exploration before training starts |
| `--collect-interval` | `100` | Gradient steps per episode |
| `--throttle_base` | `0.3` | Fixed throttle when `--fix_speed` |
| `--planning-horizon` | `15` | Imagination rollout length |
| `--augment` | off | Sim-to-real augmentations (brightness, shadow, blur, noise) |

---

## Simulator (`train_sim.py`)

### Prerequisites

1. Launch **DonkeySimMac** (`~/Downloads/DonkeySimMac/donkey_sim.app`), pick a track
2. Simulator listens on port `9091` — Python connects as TCP client

If port `9091` is stuck from a crashed run:
```bash
lsof -ti:9091 | xargs kill -9
```

### Basic run

```bash
conda activate donkeycar-dreamer
python train_sim.py --episodes 1000
```

### With human override

Human operator labels episodes in real time — no dependence on CTE telemetry. Useful when sim reward is unreliable or for collecting high-quality demonstrations.

```bash
python train_sim.py --episodes 1000 --human_override
```

Controls (keyboard, pygame window):

| Key | Action |
|-----|--------|
| `=` | Stop — off-track penalty (`reward = -1`) |
| `↑` Up | Reset — clean lap (`reward = 0`, episode ends) |
| `↓` Down | Quit training |

### Grayscale + augmentation (recommended for real-world transfer)

Train in sim with the same input format and augmentations the real car will see:

```bash
python train_sim.py --episodes 1000 --grayscale --augment
```

### Results

Saved to `results/<env>/<seed>/`:

| Path | Contents |
|------|----------|
| `rewards_<timestamp>.csv` | Per-episode reward, CTE stats, losses |
| `images/latest.png` | Real vs reconstructed grid (updates each test interval) |
| `images/ep_NNN.png` | Reconstruction snapshot per checkpoint (max 5 kept) |
| `videos/ep_NNN.mp4` | Rollout video per test episode |
| `models_NNN.pth` | Checkpoint every `--checkpoint-interval` episodes |
| `*.html` | Plotly loss curves (open in browser) |

---

## Real World (`train_real.py`)

Pi5 car drives and sends experience over ZMQ → server trains Dreamer → exports TFLite → pushes updated model back to car. The car halts between episodes waiting for new weights.

### Architecture

```
Pi5 Car                          Server (Mac/Linux)
---------                        ------------------
drive_physical_tflite.py         train_real.py
  - TFLite inference               - Dreamer training (PyTorch)
  - sends episodes via ZMQ →       - exports TFLite
  - receives updated model ←       - pushes via ZMQ
  port 5555 (experience)
  port 5556 (model)
```

### Step 1 — Bootstrap initial TFLite (server)

**From scratch (random weights):**
```bash
conda activate donkeycar-dreamer
python scripts/init_weights.py   # creates init_weights.pth + inference.tflite
scp inference.tflite car:~/pistarlab-dreamer/inference.tflite
```

**From a sim checkpoint:**
```bash
python scripts/export_pth_to_tflite.py \
    results/donkey-generated-track-v0/42/models_500.pth \
    --output inference.tflite
scp inference.tflite car:~/pistarlab-dreamer/inference.tflite
```

> Bootstrapping from a sim checkpoint gives the car a head start — it already knows roughly how to steer before seeing any real data.

### Step 2 — Pi5 dependencies (one-time)

```bash
ssh car
pip install pyzmq   # tflite-runtime numpy opencv-python donkeycar already installed
```

### Step 3 — Get server IP

```bash
# On server (Mac)
ipconfig getifaddr en0   # e.g. 192.168.1.105
```

> Set a static DHCP lease by MAC address on your router so the IP never changes between sessions.

### Step 4 — Start server trainer

```bash
conda activate donkeycar-dreamer
python train_real.py --episodes 500 --grayscale --augment
```

Server blocks on ports `5555`/`5556`, waiting for the car.

**Key real-world flags:**

| Flag | Default | Notes |
|------|---------|-------|
| `--grayscale` | `True` | Must match car drive script `--channels` |
| `--augment` | `True` | Sim-to-real augmentations on by default |
| `--push_interval` | `1` | Export + push TFLite every N episodes |
| `--checkpoint_interval` | `50` | Save `.pth` every N episodes |
| `--results-dir` | `results/real` | Output directory |
| `--smooth_weight` | `0.05` | Jitter penalty — tune based on observed oscillation |
| `--smooth_window` | `10` | Window for rolling std dev penalty |

**Bootstrap from a prior run:**
```bash
python train_real.py \
    --models results/real/models_100.pth \
    --experience-replay results/real/experience.pth \
    --episodes 500 --grayscale --augment
```

### Step 5 — Start car inference

```bash
ssh car
cd ~/pistarlab-dreamer
python scripts/drive_physical_tflite.py \
    --model inference.tflite \
    --server_ip 192.168.1.105   # replace with your server IP
```

- Press **Ctrl+C once** — end episode, send to server, wait for new model
- Press **Ctrl+C twice within 3 s** — quit

### Step 6 — Monitor training

```bash
# Live reward log
tail -f results/real/rewards_*.csv

# World model quality (open in Preview, auto-refreshes)
open results/real/images/latest.png

# Loss curves (open in browser)
open results/real/observation_loss.html
open results/real/reward_loss.html
```

The `latest.png` grid shows **real observations** (top row) vs **world model reconstructions** (bottom row) sampled from the replay buffer. Sharp reconstructions = world model has learned the track appearance.

### Real-world tuning guide

**Jittery steering:**
- Increase `--smooth_weight` (try `0.1` → `0.2`)
- Increase `--smooth_window` to penalise sustained oscillation (try `15`–`20`)
- More episodes before the policy stabilises is normal

**Car not learning:**
- Check `latest.png` — if reconstructions are blurry, world model hasn't converged yet; wait more episodes
- Reduce `--push_interval` to `2`–`3` so the car gets more data before weights update
- Ensure `--grayscale` matches between server and car script

**Off-track immediately:**
- Bootstrapping from a sim checkpoint (`--models`) gives a better starting policy than random init
- Lower `--throttle_base` (try `0.2`) for tighter tracks

**Slow training:**
- `--collect-interval 100` ≈ 3 min per episode on MPS — expected
- First `--seed-episodes 5` episodes collect data without training; server will say "Seed episode — skipping training"

### Resuming after crash

```bash
# Server
python train_real.py \
    --models results/real/models_100.pth \
    --experience-replay results/real/experience.pth \
    --episodes 500 --grayscale --augment

# Car — latest TFLite already on car, just restart
ssh car && cd ~/pistarlab-dreamer
python scripts/drive_physical_tflite.py --model inference.tflite --server_ip <server_ip>
```

---

## Project Structure

```
train_sim.py                  — simulator training entry point
train_real.py                 — real-world server training entry point
dreamer/
  agent.py                    — Dreamer agent (RSSM + SAC actor-critic)
  config.py                   — shared argparse definitions
  memory.py                   — experience replay buffer
  comms.py                    — ZMQ car ↔ server communication
  augmentations.py            — sim-to-real augmentations
  models/
    world_model.py            — TransitionModel (RSSM), Encoder, decoders
    policy.py                 — ActorModel, ValueModel
  envs/
    env.py                    — DonkeyCar, Gym, ControlSuite wrappers
    human_override.py         — keyboard labelling (Mac)
    ps4_override.py           — PS4 controller (Pi5)
  utils/
    math_utils.py             — bottle(), symlog, symexp, cal_returns
    viz.py                    — Plotly line plots, video writing
    device.py                 — CUDA/MPS/CPU detection
scripts/
  export_pth_to_tflite.py     — PyTorch checkpoint → TFLite
  drive_physical_tflite.py    — car inference loop (Pi5)
  init_weights.py             — bootstrap random-init TFLite
  baseline_report.py          — run comparison visualisation
  compare_runs.py             — CSV reward comparison
```

---

## References

1. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
2. [Mastering Atari with Discrete World Models (Dreamer v2)](https://arxiv.org/abs/2010.02193)
3. [Mastering Diverse Domains through World Models (Dreamer v3)](https://arxiv.org/abs/2301.04104)
4. [Learning Latent Dynamics for Planning from Pixels (PlaNet)](https://arxiv.org/abs/1811.04551)
5. [Donkeycar](https://www.donkeycar.com/)
