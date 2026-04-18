#!/usr/bin/env bash
# Run one ablation experiment across all 3 seeds sequentially.
#
# Usage:
#   bash scripts/run_ablation.sh baseline
#   bash scripts/run_ablation.sh no_kl_balance
#   bash scripts/run_ablation.sh no_symlog
#   bash scripts/run_ablation.sh no_return_norm
#
# Set the ablation flags in config.toml [automated] BEFORE running.
# The experiment_name in config.toml is overridden by the argument here.

set -e

NAME=${1:?"Usage: bash scripts/run_ablation.sh <experiment_name>"}

echo "========================================"
echo "  Experiment : $NAME"
echo "  Seeds      : 1  2  3"
echo "========================================"

for SEED in 1 2 3; do
    echo ""
    echo "--- Seed $SEED / 3 ---"
    conda run -n donkeycar-dreamer python train_sim_automated.py --seed "$SEED" --name "$NAME"
done

echo ""
echo "========================================"
echo "  Done: $NAME  (seeds 1 2 3)"
echo "========================================"
