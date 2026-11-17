#!/bin/bash
# Main comparison: LeWM, PLDM and SMWM, raw vs VAE skills, on the four tasks, three seeds. Extra arguments go to train.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

train() {  # train <experiment> <seed> [hydra overrides...]
    local exp="$1" seed="$2"; shift 2
    uv run python src/train.py +experiment="$exp" seed="$seed" run_id="${exp//\//_}_seed$seed" "$@"
}

for seed in 42 43 44; do
    for method in lewm pldm smwm; do
        for env in tworoom pusht cube reacher; do
            train "$method/raw_$env" "$seed" "$@"
            train "$method/vae_${env}_detach0_dec1_enc1" "$seed" "$@"
        done
    done
done
