#!/bin/bash
# Planning sweeps for train_pos_embedding.sh's checkpoints; the raw and VAE baselines are eval_main.sh's seed 42. Extra arguments go to eval_sweep.py.
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

for env in tworoom pusht cube reacher; do
    sweep "lewm/vae_${env}_detach0_dec1_enc1_pos" 42 "$@"
    sweep "lewm/vae_${env}_detach0_dec1_enc1_pos_posthocdec" 42 "$@"
done
