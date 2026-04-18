# Ablation Experiments

Automated sim training — no keyboard needed. Episodes terminate via CTE threshold.
Results saved to `results/donkey-generated-track-v0/{seed}/rewards_{experiment_name}_seed{seed}.csv`

## Run commands

Set the ablation flags in `config.toml [automated]`, then run all 3 seeds at once:

```bash
bash scripts/run_ablation.sh baseline
bash scripts/run_ablation.sh no_kl_balance
bash scripts/run_ablation.sh no_symlog
bash scripts/run_ablation.sh no_return_norm
```

Each script runs seeds 1 → 2 → 3 sequentially (one sim on port 9091).  
The `--name` and `--seed` flags are set automatically — no need to edit the toml between seeds.

To run a single seed manually:
```bash
python train_sim_automated.py --seed 1 --name baseline
```

Edit only the **ablation flags** in `config.toml [automated]` before each experiment. Only the fields in the table below change.

---

## Experiment Table

| # | `experiment_name`  | `seed`  | `kl_balance` | `symlog_rewards` | `return_norm` |
|---|--------------------|---------|--------------|------------------|---------------|
| 1 | `"baseline"`       | 1       | true         | true             | true          |
| 2 | `"baseline"`       | 2       | true         | true             | true          |
| 3 | `"baseline"`       | 3       | true         | true             | true          |
| 4 | `"no_kl_balance"`  | 1       | **false**    | true             | true          |
| 5 | `"no_kl_balance"`  | 2       | **false**    | true             | true          |
| 6 | `"no_kl_balance"`  | 3       | **false**    | true             | true          |
| 7 | `"no_symlog"`      | 1       | true         | **false**        | true          |
| 8 | `"no_symlog"`      | 2       | true         | **false**        | true          |
| 9 | `"no_symlog"`      | 3       | true         | **false**        | true          |
| 10| `"no_return_norm"` | 1       | true         | true             | **false**     |
| 11| `"no_return_norm"` | 2       | true         | true             | **false**     |
| 12| `"no_return_norm"` | 3       | true         | true             | **false**     |

**12 runs total. 20 episodes × 3 seeds per experiment ≈ 2–3 hours per experiment → ~10 hours total.**

---

## Seeds

Use **1, 2, 3**. Rationale:
- Simple and non-cherry-picked — hardest to accuse of seed shopping
- Distinct enough to capture variance (avoid 42/43/44 which are correlated by proximity)
- Standard in RL papers (Dreamer v2, DreamerV3, TD-MPC all report seed 1–3 or 1–5)

Do **not** use seed 42 as one of three — it's the config.toml default and will look like you only ran one real seed.

---

## What to Report

For each experiment, compute across 3 seeds:

| Metric | Definition |
|---|---|
| **Episodes to convergence** | First episode where reward ≥ 0.8 and stays there |
| **Mean final reward** | Mean reward over episodes 25–30 |
| **Reward std** | Std dev across seeds at episode 30 |
| **Mean episode length** | Proxy for survival time (higher = better driving) |

Plot: learning curves (reward vs episode) with shaded ±1 std band across seeds.

---

## Expected Outcomes

| Experiment | Expected effect vs baseline |
|---|---|
| `no_kl_balance` | Slower convergence, possible posterior collapse episodes 1–8 |
| `no_symlog` | Minimal effect (rewards are small integers, already well-scaled) |
| `no_return_norm` | Unstable actor loss around episodes 8–12 when reward improves |
