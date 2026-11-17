#!/bin/bash
# Observation-conditioning ablation on PushT, seed 42; dec1_enc1 is train_main.sh's run. Extra arguments go to train.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

train() {  # train <experiment> <seed> [hydra overrides...]
    local exp="$1" seed="$2"; shift 2
    uv run python src/train.py +experiment="$exp" seed="$seed" run_id="${exp//\//_}_seed$seed" "$@"
}

for exp in lewm/vae_pusht_detach0_dec0_enc0 lewm/vae_pusht_detach0_dec0_enc1 lewm/vae_pusht_detach0_dec1_enc0; do
    train "$exp" 42 "$@"
done
