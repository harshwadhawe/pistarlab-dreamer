"""
End-to-end local test for the real-world training pipeline.

Covers:
  1. Create dummy Dreamer checkpoint (no sim needed)
  2. Export checkpoint → inference.tflite via export_pth_to_tflite.py
  3. Load TFLite + run one inference step (verifies tensor names + shapes)
  4. ZMQ round-trip: fake car sends episode → mini server loop receives,
     trains, exports new TFLite, publishes back → car receives + hot-reloads

Uses ai_edge_litert (already installed) as a local stand-in for tflite_runtime
which lives on the Pi5. API is identical.

Run:
  conda activate donkeycar-dreamer
  python test_e2e_realworld.py
"""

import os, sys, subprocess, tempfile, threading, time, traceback
import numpy as np
import torch

sys.path.insert(0, '.')

BELIEF_SIZE    = 200
STATE_SIZE     = 30
ACTION_SIZE    = 2
EMBEDDING_SIZE = 1024
HIDDEN_SIZE    = 300
CHANNELS       = 1
THROTTLE_BASE  = 0.3

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

results = {}


def section(title):
    print(f'\n{"="*52}\n  {title}\n{"="*52}')


# ---------------------------------------------------------------------------
# 1. Create dummy checkpoint
# ---------------------------------------------------------------------------
section('1. Create dummy checkpoint')

from dreamer.models.world_model import TransitionModel, VisualEncoder, VisualObservationModel, RewardModel
from dreamer.models.policy import ActorModel, ValueModel

ckpt_path    = tempfile.mktemp(suffix='.pth')
tflite_path  = tempfile.mktemp(suffix='.tflite')
tflite2_path = tempfile.mktemp(suffix='.tflite')

try:
    encoder    = VisualEncoder(EMBEDDING_SIZE, channels=CHANNELS).eval()
    transition = TransitionModel(BELIEF_SIZE, STATE_SIZE, ACTION_SIZE,
                                 HIDDEN_SIZE, EMBEDDING_SIZE).eval()
    obs_model  = VisualObservationModel(BELIEF_SIZE, STATE_SIZE, EMBEDDING_SIZE,
                                        channels=CHANNELS).eval()
    reward     = RewardModel(BELIEF_SIZE, STATE_SIZE, HIDDEN_SIZE).eval()
    actor      = ActorModel(ACTION_SIZE, BELIEF_SIZE, STATE_SIZE, HIDDEN_SIZE,
                            fix_speed=True, throttle_base=THROTTLE_BASE).eval()
    value      = ValueModel(BELIEF_SIZE, STATE_SIZE, HIDDEN_SIZE).eval()

    torch.save({
        'transition_model':  transition.state_dict(),
        'observation_model': obs_model.state_dict(),
        'reward_model':      reward.state_dict(),
        'encoder':           encoder.state_dict(),
        'actor_model':       actor.state_dict(),
        'value_model':       value.state_dict(),
        'value_model2':      value.state_dict(),
        'world_optimizer':   {},
        'actor_optimizer':   {},
        'value_optimizer':   {},
    }, ckpt_path)
    print(f'Checkpoint saved → {ckpt_path}')
    results['checkpoint'] = True
except Exception:
    traceback.print_exc()
    results['checkpoint'] = False


# ---------------------------------------------------------------------------
# 2. Export to TFLite
# ---------------------------------------------------------------------------
section('2. Export checkpoint → TFLite')

try:
    cmd = [
        sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
        '--output', tflite_path,
        '--channels', str(CHANNELS),
        '--belief-size', str(BELIEF_SIZE),
        '--state-size', str(STATE_SIZE),
        '--action-size', str(ACTION_SIZE),
        '--embedding-size', str(EMBEDDING_SIZE),
        '--hidden-size', str(HIDDEN_SIZE),
        '--throttle-base', str(THROTTLE_BASE),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Filter noisy TF/JAX logs, keep our prints
    for line in proc.stdout.splitlines():
        if any(x in line for x in ['Exported', 'action:', 'new_belief:', 'new_state:', 'Converting', 'Loading']):
            print(line)
    if proc.returncode != 0:
        print(proc.stderr[-800:])
    results['export'] = proc.returncode == 0 and os.path.exists(tflite_path)
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f'TFLite size: {size_kb:.0f} KB  →  {PASS if results["export"] else FAIL}')
except Exception:
    traceback.print_exc()
    results['export'] = False


# ---------------------------------------------------------------------------
# 3. TFLite inference step (ai_edge_litert as local tflite_runtime stand-in)
# ---------------------------------------------------------------------------
section('3. TFLite inference step')

try:
    import ai_edge_litert.interpreter as tflite_mod

    interp = tflite_mod.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = {d['name']: d['index'] for d in interp.get_input_details()}
    out = {d['name']: d['index'] for d in interp.get_output_details()}

    # litert_torch uses generic names — map by shape instead
    def by_shape(details):
        return {tuple(d['shape'].tolist()): d['index'] for d in details}

    inp = by_shape(interp.get_input_details())
    out = by_shape(interp.get_output_details())
    print('Input  shapes:', list(inp.keys()))
    print('Output shapes:', list(out.keys()))

    belief  = np.zeros((1, BELIEF_SIZE),  dtype=np.float32)
    state   = np.zeros((1, STATE_SIZE),   dtype=np.float32)
    action  = np.zeros((1, ACTION_SIZE),  dtype=np.float32)
    obs     = np.zeros((1, CHANNELS, 64, 64), dtype=np.float32)

    interp.set_tensor(inp[(1, CHANNELS, 64, 64)], obs)
    interp.set_tensor(inp[(1, BELIEF_SIZE)],       belief)
    interp.set_tensor(inp[(1, STATE_SIZE)],        state)
    interp.set_tensor(inp[(1, ACTION_SIZE)],       action)
    interp.invoke()

    act_out    = interp.get_tensor(out[(1, ACTION_SIZE)])
    belief_out = interp.get_tensor(out[(1, BELIEF_SIZE)])
    state_out  = interp.get_tensor(out[(1, STATE_SIZE)])

    print(f'action:     {act_out.shape}  values: {act_out[0]}')
    print(f'new_belief: {belief_out.shape}')
    print(f'new_state:  {state_out.shape}')

    shapes_ok = (
        act_out.shape    == (1, ACTION_SIZE) and
        belief_out.shape == (1, BELIEF_SIZE) and
        state_out.shape  == (1, STATE_SIZE)
    )
    results['inference'] = shapes_ok
    print(f'Shapes correct → {PASS if shapes_ok else FAIL}')
except Exception:
    traceback.print_exc()
    results['inference'] = False


# ---------------------------------------------------------------------------
# 4. Full ZMQ round-trip: car → server (train + export) → car (reload)
# ---------------------------------------------------------------------------
section('4. ZMQ round-trip: car → server → car')

from dreamer.comms import ExperienceSender, ExperienceReceiver, ModelPublisher, ModelSubscriber

round_trip_result = {}

def mini_server():
    """Minimal server: receive 1 episode, train if buffer warm, export TFLite, publish."""
    try:
        import types
        from dreamer.agent import Dreamer

        # Build args namespace matching Dreamer's expectations
        args = types.SimpleNamespace(
            belief_size=BELIEF_SIZE, state_size=STATE_SIZE,
            action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation_size=(CHANNELS, 64, 64),
            symbolic=False, bit_depth=8,
            experience_size=100000, batch_size=10, chunk_size=10,
            collect_interval=2,
            world_lr=6e-4, actor_lr=8e-5, value_lr=8e-5,
            adam_epsilon=1e-7, grad_clip_norm=100.0,
            learning_rate_schedule=0,
            planning_horizon=5, discount=0.99, disclam=0.95,
            polyak=0.005, free_nats=1.0,
            expl_amount=0.0, with_logprob=False,
            auto_temp=False, temp=0.003,
            kl_balance=True, symlog_rewards=True, return_norm=True,
            reward_scale=10, pcont=False, pcont_scale=10,
            fix_speed=True, throttle_base=THROTTLE_BASE,
            throttle_min=0.1, throttle_max=0.5,
            angle_min=-1.0, angle_max=1.0,
            dense_act='elu', cnn_act='relu',
            augment=False, smooth_weight=0.0,
            device=torch.device('cpu'),
        )

        agent = Dreamer(args)
        # Load dummy weights so export has real structure
        ckpt = torch.load(ckpt_path, map_location='cpu')
        agent.transition_model.load_state_dict(ckpt['transition_model'])
        agent.encoder.load_state_dict(ckpt['encoder'])
        agent.actor_model.load_state_dict(ckpt['actor_model'])
        agent.observation_model.load_state_dict(ckpt['observation_model'])
        agent.reward_model.load_state_dict(ckpt['reward_model'])
        agent.value_model.load_state_dict(ckpt['value_model'])
        agent.value_model2.load_state_dict(ckpt['value_model2'])

        rx  = ExperienceReceiver(bind_ip='127.0.0.1')
        pub = ModelPublisher(bind_ip='127.0.0.1')

        round_trip_result['server_ready'] = True
        ep = rx.recv()
        T  = len(ep['rewards'])

        agent.append_episode(ep['obs'], ep['actions'], ep['rewards'], ep['dones'])

        round_trip_result['episode_received'] = ep['episode_num']

        # Train (buffer has T steps — use tiny batch/chunk for test)
        if agent.D.steps >= args.chunk_size:
            agent.update_parameters(args.collect_interval)
            round_trip_result['trained'] = True

        # Export TFLite
        cmd = [
            sys.executable, 'scripts/export_pth_to_tflite.py', ckpt_path,
            '--output', tflite2_path,
            '--channels', str(CHANNELS),
            '--belief-size', str(BELIEF_SIZE),
            '--state-size', str(STATE_SIZE),
            '--action-size', str(ACTION_SIZE),
            '--embedding-size', str(EMBEDDING_SIZE),
            '--hidden-size', str(HIDDEN_SIZE),
            '--throttle-base', str(THROTTLE_BASE),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        round_trip_result['export2'] = proc.returncode == 0

        # Publish
        time.sleep(0.4)  # let subscriber connect
        pub.publish(tflite2_path, step=agent.D.steps)
        round_trip_result['published'] = True

    except Exception:
        traceback.print_exc()


server_thread = threading.Thread(target=mini_server, daemon=True)
server_thread.start()

# Wait for server to bind
for _ in range(20):
    if round_trip_result.get('server_ready'):
        break
    time.sleep(0.1)

# Car side
try:
    sender    = ExperienceSender('127.0.0.1')
    model_sub = ModelSubscriber('127.0.0.1')
    time.sleep(0.3)

    T = 20
    fake_obs = np.random.randn(T, CHANNELS, 64, 64).astype(np.float32) * 0.1
    sender.send(
        obs=fake_obs,
        actions=np.zeros((T, ACTION_SIZE), dtype=np.float32),
        rewards=np.ones(T, dtype=np.float32),
        dones=np.array([False] * (T-1) + [True]),
        meta={'episode_num': 0, 'steps': T},
    )
    print(f'Car sent episode (T={T})')

    server_thread.join(timeout=60)

    # Poll for model update
    update = None
    for _ in range(50):
        update = model_sub.poll()
        if update:
            break
        time.sleep(0.2)

    if update:
        # Simulate hot-reload
        tmp = '/tmp/test_reload.tflite'
        with open(tmp, 'wb') as f:
            f.write(update['model_bytes'])
        interp2 = tflite_mod.Interpreter(model_path=tmp)
        interp2.allocate_tensors()
        print(f'Car hot-reloaded model — server step {update["step"]}, '
              f'size {len(update["model_bytes"])/1024:.0f} KB')
        round_trip_result['car_reloaded'] = True

except Exception:
    traceback.print_exc()

results['round_trip'] = all([
    round_trip_result.get('episode_received') == 0,
    round_trip_result.get('trained'),
    round_trip_result.get('export2'),
    round_trip_result.get('published'),
    round_trip_result.get('car_reloaded'),
])

print(f'\nRound-trip details: {round_trip_result}')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section('RESULTS')
checks = [
    ('Checkpoint creation', results.get('checkpoint')),
    ('TFLite export',       results.get('export')),
    ('TFLite inference',    results.get('inference')),
    ('ZMQ round-trip',      results.get('round_trip')),
]
for name, ok in checks:
    status = PASS if ok else FAIL
    print(f'  {name:<25} {status}')

all_pass = all(ok for _, ok in checks)
print(f'\n{"ALL PASS" if all_pass else "SOME FAILED"}\n')

# Cleanup
for p in [ckpt_path, tflite_path, tflite2_path]:
    try: os.unlink(p)
    except: pass

sys.exit(0 if all_pass else 1)
