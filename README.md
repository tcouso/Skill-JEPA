# Skill-JEPA

JEPA world models learn from pixels to predict the outcome of actions in a latent space, and a sampling-based planner uses them to reach goals. Standard planners draw candidate actions from a Gaussian, but the model has only seen the structured, state-dependent actions of its training data, so its predictions for those candidates are unreliable and planning fails more often as goals move farther away.

Skill-JEPA learns an action space aligned with the planner. A state-conditioned variational autoencoder encodes short action sequences into latent skills with a standard Gaussian prior and decodes them back into actions. The world model predicts the effect of skills instead of raw actions, and the planner searches over skills starting from that same Gaussian. It is a single module trained end-to-end with the world model; the planner is unchanged, with no subgoals, hierarchy, or hand-designed skills.

Across three JEPA world models (LeWM, PLDM, SMWM) and four tasks (Two-Rooms, Push-T, Cube, Reacher), average success on goals 80 steps away rises from 42% with raw actions to 62%, and the gains hold across four planners (CEM, iCEM, MPPI, Predictive Sampling). On AntMaze, success on goals two maze cells away rises from 0.4% to 38%.

## Setup

Requires Python ≥ 3.12, [uv](https://github.com/astral-sh/uv), an NVIDIA GPU, and `tar`/`zstd` for dataset extraction.

```bash
uv sync
uv run python src/download_data.py   # all five datasets -> data/datasets/
```

TwoRoom, PushT, Cube and Reacher are the [le-wm datasets](https://huggingface.co/collections/quentinll/lewm); AntMaze is H-JEPA's Visual AntMaze medium training set.

To run without wandb, `export WANDB_MODE=disabled`: training writes to `checkpoints/<run_id>/`, evals to `results/`, and the figure scripts read from there. Otherwise runs log to your own wandb account (`uv run wandb login`), and the figure scripts pull from your default entity.

## Reproducing the results

Each script loops over the configs and seeds of one result and forwards extra arguments as Hydra overrides.

| Result | Train | Evaluate |
|---|---|---|
| Main comparison | `scripts/train_main.sh` | `scripts/eval_main.sh` |
| AntMaze | `scripts/train_antmaze.sh` | `scripts/eval_antmaze.sh` |
| Latent-action analysis | `scripts/train_main.sh`, `scripts/train_ae.sh` | `scripts/eval_main.sh`, `scripts/eval_latent_actions.sh` |
| Planners | `scripts/train_main.sh` | `scripts/eval_planners.sh` |
| Positional embedding | `scripts/train_pos_embedding.sh` | `scripts/eval_pos_embedding.sh` |
| Observation conditioning | `scripts/train_obs_conditioning.sh` | `scripts/eval_obs_conditioning.sh` |

Then build every figure and table into `results/figures/`:

```bash
scripts/make_figures.sh
```

A single run:

```bash
uv run python src/train.py +experiment=lewm/vae_pusht_detach0_dec1_enc1 seed=42 run_id=my_run
uv run python src/eval_sweep.py +experiment=lewm/vae_pusht_detach0_dec1_enc1 ++ckpt_path=checkpoints/my_run/last.ckpt
```

## Configs

Experiments live in `config/experiment/{lewm,pldm,smwm}/`, one file per trained model. `raw_*` plans over raw actions, `vae_*` over skills, `ae_*` over skills without the KL term. In the names, `dec<0|1>`/`enc<0|1>` set whether the skill decoder/encoder see the state, `detach1` stops gradients between world model and skill autoencoder, `_pos` adds positional embeddings to the skill encoder, and `_posthocdec` trains the decoder on detached skills.
