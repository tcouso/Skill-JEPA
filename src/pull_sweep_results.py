"""Collects eval-sweep tables into one DataFrame and plots them.

Reads results/sweeps/ when WANDB_MODE=disabled, else pulls Sweep_* runs from your default wandb entity.

    uv run python src/pull_sweep_results.py
    uv run python src/pull_sweep_results.py --run-ids uyx0akah gupn8aai   # wandb only
    uv run python src/pull_sweep_results.py --output-dir results/figures/main --no-pull   # replot the CSV already there
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
import wandb

from src.config import wandb_enabled
from src.viz.sweep_plots import (
    WHITENED_SUFFIX,
    make_execution_plots,
    make_iteration0_kl_table,
    make_planner_delta_plots,
    make_plots,
)

PROJECT = "SkillJEPA_Eval"
OUTPUT_DIR = Path("results/sweep_analysis")
DEFAULT_SEED = 42
DEFAULT_SOLVER = "CEMSolver"

_ACTION_ENCODER_TO_MODE = {"none": "raw", "vae": "vae", "ae": "ae"}


def _derive_mode_and_frameskip(name: str, config: dict) -> tuple[str, int]:
    encoder = config.get("action_encoder")
    if encoder not in _ACTION_ENCODER_TO_MODE:
        raise ValueError(f"{name}: unrecognized action_encoder {encoder!r}")
    return _ACTION_ENCODER_TO_MODE[encoder], config["dataset"]["frameskip"]


_ABLATION_TAG = re.compile(r"detach\d_dec\d_enc\d(?:_pos)?(?:_posthocdec)?")


def _derive_base_label(name: str, config: dict, mode: str) -> str:
    """The checkpoint's own identity: base method and mode, plus its ablation tag."""
    regularizer = config.get("regularizer", "sigreg")
    prefix = "lewm/" if regularizer == "sigreg" else f"{regularizer}/"
    ablation = _ABLATION_TAG.search(name)
    return f"{prefix}{mode} {ablation.group()}" if ablation else f"{prefix}{mode}"


_ENV_SLUGS = {
    "ogbcube": "cube",
    "ogbantmaze": "antmaze",
    "reacherdmcontrol": "reacher",
}


def _derive_env(name: str, config: dict) -> str:
    """'swm/PushT-v1' -> 'pusht'; 'swm/ReacherDMControl-v0' -> 'reacher'."""
    env_name = config.get("world", {}).get("env_name")
    if not env_name:
        raise ValueError(f"{name}: no world.env_name in logged config")
    slug = re.sub(r"-v\d+$", "", env_name.split("/")[-1]).lower()
    return _ENV_SLUGS.get(slug, slug)


# JSON nulls make pandas type a column as object, which the seed aggregation rejects.
_NUMERIC_COLUMNS = (
    "success_rate", "n_success", "eval_time_s", "goal_offset_steps", "goal_cell_distance",
    "plan_horizon", "eval_budget", "cem_num_samples", "cem_topk",
)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


_POINT_KEY = [
    "env", "frameskip", "cost_mode", "loop_mode", "label", "seed",
    "goal_offset_steps", "plan_horizon", "cem_num_samples",
]


def keep_newest_per_point(combined: pd.DataFrame) -> pd.DataFrame:
    """One row per plotted point, newest run winning."""
    ordered = combined.sort_values("created_at", kind="stable")
    kept = ordered.drop_duplicates(subset=_POINT_KEY, keep="last")
    dropped = ordered.loc[~ordered.index.isin(kept.index), "checkpoint"].value_counts()
    for checkpoint, count in dropped.items():
        print(f"[dedup] superseded at a shared label: {checkpoint} ({count} rows)")
    return kept.sort_index()


SWEEPS_DIR = Path("results/sweeps")


def download_table(run, table_ref: dict, root: Path) -> pd.DataFrame:
    file = run.file(table_ref["path"]).download(root=str(root), replace=True)
    with open(file.name) as f:
        table_json = json.load(f)
    return pd.DataFrame(table_json["data"], columns=table_json["columns"])


def wandb_project_path() -> tuple[wandb.Api, str]:
    api = wandb.Api()
    return api, f"{api.default_entity}/{PROJECT}"


def tag_sweep_frame(df: pd.DataFrame, checkpoint: str, config: dict, created_at: str) -> pd.DataFrame:
    mode, frameskip = _derive_mode_and_frameskip(checkpoint, config)
    df["checkpoint"] = checkpoint
    df["mode"] = mode
    df["frameskip"] = frameskip
    df["env"] = _derive_env(checkpoint, config)
    df["seed"] = config.get("train_seed", DEFAULT_SEED)
    df["base_label"] = _derive_base_label(checkpoint, config, mode)
    df["ckpt_path"] = config["ckpt_path"]
    df["created_at"] = created_at
    return df


def pull_wandb_sweeps(run_ids: list[str] | None, raw_dir: Path) -> list[pd.DataFrame]:
    """Every Sweep_* table, or just `run_ids`."""
    api, path = wandb_project_path()
    if run_ids:
        sweep_runs = [api.run(f"{path}/{rid}") for rid in run_ids]
    else:
        sweep_runs = [r for r in api.runs(path) if r.name.startswith("Sweep_")]

    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for run in sweep_runs:
        table_ref = run.summary.get("sweep_results")
        if table_ref is None:
            print(f"[skip] {run.name}: no sweep_results table logged")
            continue

        df = download_table(run, table_ref, raw_dir)
        frames.append(tag_sweep_frame(df, run.name.removeprefix("Sweep_"), dict(run.config), str(run.created_at)))
    return frames


def read_local_sweeps() -> list[pd.DataFrame]:
    """Every results/sweeps/<checkpoint>/ that eval_sweep.py wrote in local mode."""
    frames = []
    for info_path in sorted(SWEEPS_DIR.rglob("run.json")):
        table_path = info_path.parent / "sweep_results.csv"
        if not table_path.exists():
            print(f"[skip] {info_path.parent}: no sweep_results.csv")
            continue
        info = json.loads(info_path.read_text())
        checkpoint = str(info_path.parent.relative_to(SWEEPS_DIR))
        frames.append(tag_sweep_frame(pd.read_csv(table_path), checkpoint, info, info["created_at"]))
    return frames


def collect_sweep_tables(
    run_ids: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
    match: str | None = None,
) -> pd.DataFrame:
    """All sweep tables in one checkpoint-tagged frame, from wandb or from results/sweeps/."""
    if wandb_enabled():
        frames = pull_wandb_sweeps(run_ids, output_dir / "_raw")
    else:
        frames = read_local_sweeps()
    if match:
        frames = [f for f in frames if re.search(match, f["checkpoint"].iloc[0])]
    if not frames:
        raise RuntimeError("No sweep tables found")

    combined = coerce_numeric_columns(pd.concat(frames, ignore_index=True))
    combined["whiten_skill_space"] = combined["whiten_skill_space"].astype(bool)
    combined["label"] = (
        combined["base_label"]
        + combined["whiten_skill_space"].map(lambda w: WHITENED_SUFFIX if w else "")
        + combined["solver"].map(lambda s: "" if s == DEFAULT_SOLVER else f" ({s})")
    )
    combined = keep_newest_per_point(combined)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "combined_sweep_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Collected {len(frames)} sweep runs, {len(combined)} rows -> {csv_path}")
    return combined


def collect_iteration0_kl(combined: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """KL(skill prior fitted at training || N(0, I)) per latent checkpoint, read from the checkpoint."""
    latent = combined[combined["mode"].isin(["ae", "vae"]) & ~combined["whiten_skill_space"]]
    rows = []
    for checkpoint in latent.drop_duplicates("ckpt_path").itertuples():
        state_dict = torch.load(checkpoint.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        mu = next(v for k, v in state_dict.items() if k.endswith("skill_bottleneck.skill_prior_mu")).double()
        var = next(v for k, v in state_dict.items() if k.endswith("skill_bottleneck.skill_prior_var")).double()
        kl = 0.5 * (mu**2 + var - torch.log(var) - 1.0).sum()
        rows.append({"env": checkpoint.env, "mode": checkpoint.mode, "label": checkpoint.base_label,
                     "seed": checkpoint.seed, "ckpt_path": checkpoint.ckpt_path, "kl": kl.item()})

    frame = pd.DataFrame(rows).sort_values(["env", "mode", "seed"])
    csv_path = output_dir / "iteration0_kl.csv"
    frame.to_csv(csv_path, index=False)
    print(f"Iteration-0 KL for {len(frame)} latent checkpoints -> {csv_path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-ids", nargs="+", default=None,
        help="wandb mode: pull exactly these run IDs instead of every Sweep_* run.",
    )
    parser.add_argument(
        "--match", default=None,
        help="Keep only checkpoints whose name matches this regex, e.g. '^lewm/'.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Write the frame and plots here instead of the default, so a "
             "restricted pull does not overwrite the full-history one.",
    )
    parser.add_argument(
        "--no-pull", action="store_true",
        help="Replot from <output-dir>/combined_sweep_results.csv instead of collecting the sweeps again.",
    )
    parser.add_argument(
        "--planner-delta", action="store_true",
        help="Also plot vae - raw success per planner.",
    )
    parser.add_argument(
        "--iteration0-kl", action="store_true",
        help="Also tabulate KL(fitted skill prior || N(0, I)) per latent checkpoint, read from its checkpoint file.",
    )
    parser.add_argument(
        "--execution", action="store_true",
        help="Also plot predicted-vs-executed error on dataset plans and CEM first-iteration candidates, "
             "reading <output-dir>/dataset_by_step.csv and iteration0_by_step.csv from pull_component_evals.",
    )
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    if args.no_pull:
        frame = pd.read_csv(out / "combined_sweep_results.csv")
    else:
        frame = collect_sweep_tables(run_ids=args.run_ids, output_dir=out, match=args.match)
    make_plots(frame, out)
    if args.planner_delta:
        make_planner_delta_plots(frame, out)
    if args.iteration0_kl:
        make_iteration0_kl_table(collect_iteration0_kl(frame, out), out)
    if args.execution:
        make_execution_plots(pd.read_csv(out / "dataset_by_step.csv"), pd.read_csv(out / "iteration0_by_step.csv"), out)


if __name__ == "__main__":
    main()
