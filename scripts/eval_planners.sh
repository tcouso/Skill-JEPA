#!/bin/bash
# Planner ablation: iCEM, MPPI and Predictive Sampling on LeWM raw vs VAE checkpoints from train_main.sh (CEM is eval_main.sh). Extra arguments go to eval_sweep.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GRID=(
    "evaluation.sweep_conditions=[terminal_open_loop]"
    "evaluation.sweep_goal_offsets=[20,40,60,80]"
    "evaluation.num_eval=100"
)

sweep() {  # sweep <experiment> <seed> [hydra overrides...]
    local exp="$1" seed="$2"; shift 2
    uv run python src/eval_sweep.py +experiment="$exp" ++ckpt_path="checkpoints/${exp//\//_}_seed$seed/last.ckpt" "${GRID[@]}" "$@"
}

for seed in 42 43 44; do
    for solver in icem mppi predictive_sampling; do
        for env in tworoom pusht cube reacher; do
            for exp in "lewm/raw_$env" "lewm/vae_${env}_detach0_dec1_enc1"; do
                sweep "$exp" "$seed" solver="$solver" evaluation.eval_tag="${exp}_$solver" "$@"
            done
        done
    done
done
