#!/bin/bash
# Planning sweeps for train_antmaze.sh's checkpoints over goal cell distances 1-4 (config/env/antmaze.yaml). Extra arguments go to eval_sweep.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

sweep() {  # sweep <experiment> <seed> [hydra overrides...]
    local exp="$1" seed="$2"; shift 2
    uv run python src/eval_sweep.py +experiment="$exp" ++ckpt_path="checkpoints/${exp//\//_}_seed$seed/last.ckpt" \
        "evaluation.sweep_conditions=[terminal_open_loop]" "evaluation.num_eval=100" "$@"
}

for seed in 42 43 44; do
    for method in lewm pldm smwm; do
        sweep "$method/raw_antmaze" "$seed" "$@"
        sweep "$method/vae_antmaze_detach0_dec1_enc1" "$seed" "$@"
    done
done
