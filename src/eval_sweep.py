"""Planning eval: one checkpoint over a grid of cells; an unset sweep axis contributes the config's single value."""

import os

os.environ["MUJOCO_GL"] = "egl"  # must precede any mujoco-touching import

import csv
import time
import traceback

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.config import JEPAConfig, reject_stray_experiment, wandb_enabled
from src.evaluation.checkpoint import (
    apply_eval_overrides,
    derive_tag,
    prepare_system,
    results_dir,
    write_run_info,
)
from src.evaluation.mpc_eval import eval_budget, plan_horizon, run_mpc_eval
from src.training import SkillJEPASystem

LOOP_MODES = {"closed_loop": 1, "open_loop": None}

CELL_COLUMNS = [
    "goal_offset_steps", "goal_cell_distance", "plan_horizon", "eval_budget", "cost_mode", "loop_mode",
    "solver", "cem_num_samples", "cem_topk", "whiten_skill_space",
]
METRIC_COLUMNS = ["success_rate", "n_success"]
TABLE_COLUMNS = CELL_COLUMNS + METRIC_COLUMNS + ["eval_time_s", "error"]
EPISODE_COLUMNS = CELL_COLUMNS + ["env_index", "dataset_episode", "start_step", "success", "steps_to_done"]


def resolve_conditions(tokens: list[str] | None, config: JEPAConfig) -> list[tuple[str, str]]:
    """(cost_mode, loop_mode) pairs from "<cost_mode>_<loop_mode>" tokens."""
    if not tokens:
        loop_mode = {v: k for k, v in LOOP_MODES.items()}[config.action.macro_commitment]
        return [(config.evaluation.cost_mode, loop_mode)]
    return [(cost_mode, loop_mode) for cost_mode, _, loop_mode in (t.partition("_") for t in tokens)]


def build_sweep_grid(
    goal_offsets: list[int],
    goal_cell_distances: list[int | None],
    conditions: list[tuple[str, str]],
) -> list[dict]:
    """One cell per (goal offset, cell distance) pair and condition; the two lists are paired by position."""
    return [
        {
            "goal_offset_steps": goal_offset,
            "goal_cell_distance": goal_cell_distance,
            "cost_mode": cost_mode,
            "loop_mode": loop_mode,
        }
        for goal_offset, goal_cell_distance in zip(goal_offsets, goal_cell_distances, strict=True)
        for cost_mode, loop_mode in conditions
    ]


def apply_cell(cell: dict, system: SkillJEPASystem, config: JEPAConfig, cfg: DictConfig) -> dict:
    """Point config, solver and model at `cell`; returns its full set of table columns."""
    config.evaluation.goal_offset_steps = cell["goal_offset_steps"]
    config.evaluation.goal_cell_distance = cell["goal_cell_distance"]
    config.evaluation.cost_mode = cell["cost_mode"]
    config.action.macro_commitment = LOOP_MODES[cell["loop_mode"]]
    apply_eval_overrides(system, config)

    return {
        **cell,
        "plan_horizon": plan_horizon(config),
        "eval_budget": eval_budget(config),
        "solver": cfg.solver._target_.rsplit(".", 1)[-1],
        "cem_num_samples": cfg.solver.get("num_samples"),
        "cem_topk": cfg.solver.get("topk"),
        "whiten_skill_space": config.evaluation.whiten_skill_space,
    }


def append_episode_rows(path, rows: list[dict]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def run_sweep(system: SkillJEPASystem, config: JEPAConfig, cfg: DictConfig, tag: str) -> None:
    """One MPC eval per cell, the table re-logged after each so a timeout keeps finished rows."""
    goal_offsets = config.evaluation.sweep_goal_offsets or [config.evaluation.goal_offset_steps]
    goal_cell_distances = (
        config.evaluation.sweep_goal_cell_distances
        or [config.evaluation.goal_cell_distance] * len(goal_offsets)
    )
    grid = build_sweep_grid(
        goal_offsets,
        goal_cell_distances,
        resolve_conditions(config.evaluation.sweep_conditions, config),
    )
    out = results_dir(tag)
    episodes_path = out / "episodes.csv"
    episodes_path.unlink(missing_ok=True)
    sweep_rows = []
    # wandb freezes a non-MUTABLE table at its first log(); later add_data() no-ops.
    table = wandb.Table(columns=TABLE_COLUMNS, log_mode="MUTABLE")
    episode_table = wandb.Table(columns=EPISODE_COLUMNS, log_mode="MUTABLE")
    print(f"[eval_sweep] {tag}: {len(grid)} cells")

    for i, cell in enumerate(grid):
        columns = apply_cell(cell, system, config, cfg)
        print(f"[eval_sweep] ({i + 1}/{len(grid)}) {columns}")

        start_time = time.time()
        metrics, episode_rows, contact_sheets, error = {}, [], [], ""
        try:
            with torch.no_grad():
                metrics, episode_rows, contact_sheets = run_mpc_eval(
                    system.jepa, config, cfg, render=config.evaluation.render
                )
        except Exception:
            error = traceback.format_exc(limit=3)
            print(f"[eval_sweep] cell {i + 1} failed:\n{error}")

        row = {
            **columns,
            **{column: metrics.get(column) for column in METRIC_COLUMNS},
            "eval_time_s": time.time() - start_time,
            "error": error,
        }
        episode_rows = [{**columns, **r} for r in episode_rows]
        if episode_rows:
            append_episode_rows(episodes_path, episode_rows)
        cell_key = "_".join(str(cell[k]) for k in ("goal_offset_steps", "cost_mode", "loop_mode"))

        if wandb_enabled():
            table.add_data(*[row[column] for column in TABLE_COLUMNS])
            for episode_row in episode_rows:
                episode_table.add_data(*[episode_row[column] for column in EPISODE_COLUMNS])
            wandb.log({"sweep_results": table, "episode_results": episode_table})
            for j, sheet in enumerate(contact_sheets):
                wandb.log({f"render/{cell_key}/contact_sheet_{j}": wandb.Image(sheet)})
            continue

        sweep_rows.append({column: row[column] for column in TABLE_COLUMNS})
        pd.DataFrame(sweep_rows).to_csv(out / "sweep_results.csv", index=False)
        for j, sheet in enumerate(contact_sheets):
            sheet_dir = out / "contact_sheets"
            sheet_dir.mkdir(exist_ok=True)
            sheet.save(sheet_dir / f"{cell_key}_{j:02d}.png")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    reject_stray_experiment(cfg)
    config: JEPAConfig = OmegaConf.to_object(cfg)
    system = prepare_system(cfg, config)

    train_seed = system.jepa.config.seed
    tag = f"{derive_tag(config)}_seed{train_seed}"
    if wandb_enabled():
        wandb.init(
            project="SkillJEPA_Eval",
            name=f"Sweep_{tag}",
            config={**OmegaConf.to_container(cfg, resolve=True), "train_seed": train_seed},
        )
    else:
        write_run_info(results_dir(tag), cfg, train_seed)
    run_sweep(system, config, cfg, tag)
    wandb.finish()


if __name__ == "__main__":
    run()
