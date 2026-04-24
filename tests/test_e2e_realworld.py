"""
End-to-end local test for the real-world training pipeline.

Covers:
  1. Create dummy Dreamer checkpoint (no sim needed)
  2. Export checkpoint → inference.tflite via export_pth_to_tflite.py
  3. Load TFLite + run one inference step (verifies tensor names + shapes)
  4. File-based comms roundtrip: sender writes npz + sentinel, watcher detects
     new model via step.txt — verifies uint8 encoding and discard sentinel.

Uses ai_edge_litert (already installed) as a local stand-in for tflite_runtime
which lives on the Pi5. API is identical.

Run:
  conda activate donkeycar-dreamer
  python test_e2e_realworld.py
"""

import os, sys, subprocess, tempfile, traceback
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

ckpt_path   = tempfile.mktemp(suffix='.pth')
tflite_path = tempfile.mktemp(suffix='.tflite')

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
# 4. File-based comms roundtrip (rsync protocol, local file I/O only)
# ---------------------------------------------------------------------------
section('4. File-based comms roundtrip (rsync protocol)')

import dreamer.comms as _comms_mod

_roundtrip = {}
try:
    with tempfile.TemporaryDirectory() as _tmpdir:
        _outbox = os.path.join(_tmpdir, 'outbox')
        _inbox  = os.path.join(_tmpdir, 'inbox')
        os.makedirs(_outbox); os.makedirs(_inbox)

        _orig_out, _orig_in = _comms_mod.PI_OUTBOX, _comms_mod.PI_INBOX
        _comms_mod.PI_OUTBOX = _outbox
        _comms_mod.PI_INBOX  = _inbox

        try:
            T        = 20
            fake_obs = np.clip(
                np.random.randn(T, CHANNELS, 64, 64).astype(np.float32) * 0.1,
                -0.5, 0.5,
            )

            # Pi sender writes episode as uint8-compressed npz + sentinel
            sender = _comms_mod.RsyncExperienceSender()
            sender.send(
                obs=fake_obs,
                actions=np.zeros((T, ACTION_SIZE), dtype=np.float32),
                rewards=np.ones(T, dtype=np.float32),
                dones=np.array([False] * (T - 1) + [True]),
                meta={'episode_num': 1},
            )

            assert os.path.exists(os.path.join(_outbox, 'ep_0001.npz')),   'npz missing'
            assert os.path.exists(os.path.join(_outbox, 'ep_0001.ready')), 'sentinel missing'

            data    = np.load(os.path.join(_outbox, 'ep_0001.npz'))
            obs_rt  = _comms_mod._uint8_to_obs(data['obs'])
            max_err = float(np.abs(obs_rt - fake_obs).max())
            assert obs_rt.shape == fake_obs.shape
            assert max_err < 0.003, f'uint8 roundtrip error: {max_err:.4f}'
            _roundtrip['episode_saved'] = True
            print(f'Episode saved: npz+sentinel present, obs roundtrip err {max_err:.4f}')

            # Discard sentinel
            sender.send_discard(2)
            assert os.path.exists(os.path.join(_outbox, 'ep_0002.discard')), 'discard missing'
            _roundtrip['discard_sent'] = True
            print('Discard sentinel written correctly')

            # Simulate server pushing model into Pi inbox
            with open(os.path.join(_inbox, 'latest.tflite'), 'wb') as f:
                f.write(b'MOCK_TFLITE_BYTES')
            with open(os.path.join(_inbox, 'step.txt'), 'w') as f:
                f.write('42')

            watcher = _comms_mod.RsyncModelWatcher()
            update  = watcher.poll()
            assert update is not None,                             'watcher returned None'
            assert update['step'] == 42,                           f'wrong step: {update["step"]}'
            assert update['model_bytes'] == b'MOCK_TFLITE_BYTES', 'wrong bytes'
            _roundtrip['model_received'] = True
            print(f'Model received: step={update["step"]}, size={len(update["model_bytes"])} B')

        finally:
            _comms_mod.PI_OUTBOX = _orig_out
            _comms_mod.PI_INBOX  = _orig_in

except Exception:
    traceback.print_exc()

results['round_trip'] = all([
    _roundtrip.get('episode_saved'),
    _roundtrip.get('discard_sent'),
    _roundtrip.get('model_received'),
])
print(f'\nRound-trip details: {_roundtrip}')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section('RESULTS')
checks = [
    ('Checkpoint creation', results.get('checkpoint')),
    ('TFLite export',       results.get('export')),
    ('TFLite inference',    results.get('inference')),
    ('File comms roundtrip', results.get('round_trip')),
]
for name, ok in checks:
    status = PASS if ok else FAIL
    print(f'  {name:<25} {status}')

all_pass = all(ok for _, ok in checks)
print(f'\n{"ALL PASS" if all_pass else "SOME FAILED"}\n')

# Cleanup
for p in [ckpt_path, tflite_path]:
    try: os.unlink(p)
    except: pass

sys.exit(0 if all_pass else 1)
