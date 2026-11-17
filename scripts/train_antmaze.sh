#!/bin/bash
# AntMaze: LeWM, PLDM and SMWM, raw vs VAE skills, three seeds. Extra arguments go to train.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

train() {  # train <experiment> <seed> [hydra overrides...]
    local exp="$1" seed="$2"; shift 2
    uv run python src/train.py +experiment="$exp" seed="$seed" run_id="${exp//\//_}_seed$seed" "$@"
}

for seed in 42 43 44; do
    for method in lewm pldm smwm; do
        train "$method/raw_antmaze" "$seed" "$@"
        train "$method/vae_antmaze_detach0_dec1_enc1" "$seed" "$@"
    done
done
