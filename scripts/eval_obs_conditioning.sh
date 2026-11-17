#!/bin/bash
# Planning sweeps for train_obs_conditioning.sh's checkpoints. Extra arguments go to eval_sweep.py.
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

for exp in lewm/vae_pusht_detach0_dec0_enc0 lewm/vae_pusht_detach0_dec0_enc1 lewm/vae_pusht_detach0_dec1_enc0; do
    sweep "$exp" 42 "$@"
done
