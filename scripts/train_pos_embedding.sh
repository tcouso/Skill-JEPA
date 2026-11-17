#!/bin/bash
# Positional-embedding ablation, seed 42: VAE skills with an encoder positional embedding, and with a post-hoc decoder. Extra arguments go to train.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

train() {  # train <experiment> <seed> [hydra overrides...]
    local exp="$1" seed="$2"; shift 2
    uv run python src/train.py +experiment="$exp" seed="$seed" run_id="${exp//\//_}_seed$seed" "$@"
}

for env in tworoom pusht cube reacher; do
    train "lewm/vae_${env}_detach0_dec1_enc1_pos" 42 "$@"
    train "lewm/vae_${env}_detach0_dec1_enc1_pos_posthocdec" 42 "$@"
done
