#!/bin/bash
# Every figure and table of the paper from the eval outputs, into results/figures/<group>/. Runs in the same mode (local or wandb) as the evals.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
OUT=results/figures
ENVS="(tworoom|pusht|cube|reacher)"

uv run python src/pull_sweep_results.py --output-dir "$OUT/main" \
    --match "^(lewm|pldm|smwm)/(raw|vae)_$ENVS(_detach0_dec1_enc1)?_seed\d+$"

uv run python src/pull_sweep_results.py --output-dir "$OUT/antmaze" \
    --match "^(lewm|pldm|smwm)/(raw|vae)_antmaze(_detach0_dec1_enc1)?_seed\d+$"

uv run python src/pull_component_evals.py --output-dir "$OUT/latent_actions"
uv run python src/pull_sweep_results.py --output-dir "$OUT/latent_actions" \
    --match "^lewm/(raw|vae|ae)_$ENVS(_detach0_dec1_enc1)?(_whiten)?_seed\d+$" \
    --execution --iteration0-kl

uv run python src/pull_sweep_results.py --output-dir "$OUT/planners" --planner-delta \
    --match "^lewm/(raw|vae)_$ENVS(_detach0_dec1_enc1)?(_icem|_mppi|_predictive_sampling)?_seed\d+$"

uv run python src/pull_sweep_results.py --output-dir "$OUT/pos_embedding" \
    --match "^lewm/(raw_$ENVS|vae_${ENVS}_detach0_dec1_enc1(_pos|_pos_posthocdec)?)_seed42$"

uv run python src/pull_sweep_results.py --output-dir "$OUT/obs_conditioning" \
    --match "^lewm/vae_pusht_detach0_dec[01]_enc[01]_seed42$"

uv run python src/viz/task_frames.py
