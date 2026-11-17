#!/bin/bash
# Latent-action analysis: planning sweeps for the AE checkpoints and component evals for LeWM raw, VAE and AE. Needs train_main.sh and train_ae.sh. Extra arguments go to both eval scripts.
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

components() {  # components <experiment> <seed> [hydra overrides...]
    local exp="$1" seed="$2"; shift 2
    uv run python src/eval_components.py +experiment="$exp" ++ckpt_path="checkpoints/${exp//\//_}_seed$seed/last.ckpt" "$@"
}

for seed in 42 43 44; do
    for env in tworoom pusht cube reacher; do
        sweep "lewm/ae_${env}_detach0_dec1_enc1" "$seed" "$@"
        for exp in "lewm/raw_$env" "lewm/vae_${env}_detach0_dec1_enc1" "lewm/ae_${env}_detach0_dec1_enc1"; do
            components "$exp" "$seed" "$@"
        done
    done
done
