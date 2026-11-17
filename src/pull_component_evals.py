"""Collects the component evals' action-encoder metrics into <output-dir>/component_evals.csv, and their per-step tables into <output-dir>/dataset_by_step.csv and <output-dir>/iteration0_by_step.csv.

Reads results/components/ when WANDB_MODE=disabled, else pulls Components_* runs from your default wandb entity.

    uv run python src/pull_component_evals.py --output-dir results/figures/latent_actions
    uv run python src/pull_component_evals.py --run-ids abc123 def456   # wandb only
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import wandb_enabled
from src.pull_sweep_results import (
    DEFAULT_SEED,
    _derive_base_label,
    _derive_env,
    _derive_mode_and_frameskip,
    download_table,
    wandb_project_path,
)

OUTPUT_DIR = Path("results")
COMPONENTS_DIR = Path("results/components")
TABLE_KEYS = {
    "dataset/error_by_step": "dataset_by_step.csv",
    "iteration0/error_by_step": "iteration0_by_step.csv",
}


def is_summary_metric(key: str) -> bool:
    return key.startswith(("action_spectrum/", "action_recon/"))


def identity_of(name: str, config: dict) -> dict:
    mode, frameskip = _derive_mode_and_frameskip(name, config)
    return {
        "run": name,
        "env": _derive_env(name, config),
        "mode": mode,
        "label": _derive_base_label(name, config, mode),
        "seed": config.get("train_seed", DEFAULT_SEED),
        "frameskip": frameskip,
    }


def pull_wandb_components(run_ids: list[str] | None, raw_dir: Path) -> tuple[list[dict], dict[str, list[pd.DataFrame]]]:
    api, path = wandb_project_path()
    if run_ids:
        runs = [api.run(f"{path}/{rid}") for rid in run_ids]
    else:
        runs = [r for r in api.runs(path) if (r.name or "").startswith("Components_")]

    rows, tables = [], {key: [] for key in TABLE_KEYS}
    for run in runs:
        identity = identity_of(run.name, dict(run.config))
        rows.append({**identity, **{k: v for k, v in run.summary.items() if is_summary_metric(k)}})
        for key in TABLE_KEYS:
            table_ref = run.summary.get(key)
            if table_ref is not None:
                tables[key].append(download_table(run, table_ref, raw_dir).assign(**identity))
    return rows, tables


def read_local_components() -> tuple[list[dict], dict[str, list[pd.DataFrame]]]:
    rows, tables = [], {key: [] for key in TABLE_KEYS}
    for info_path in sorted(COMPONENTS_DIR.rglob("run.json")):
        run_dir = info_path.parent
        identity = identity_of(str(run_dir.relative_to(COMPONENTS_DIR)), json.loads(info_path.read_text()))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        rows.append({**identity, **{k: v for k, v in metrics.items() if is_summary_metric(k)}})
        for key in TABLE_KEYS:
            table_path = run_dir / f"{key.replace('/', '_')}.csv"
            if table_path.exists():
                tables[key].append(pd.read_csv(table_path).assign(**identity))
    return rows, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", nargs="+", default=None, help="wandb mode: pull exactly these run IDs instead of every Components_* run.")
    parser.add_argument("--output-dir", default=None, help="Write here instead of results/.")
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    rows, tables = pull_wandb_components(args.run_ids, out / "_raw") if wandb_enabled() else read_local_components()
    pd.DataFrame(rows).to_csv(out / "component_evals.csv", index=False)
    print(f"{len(rows)} runs -> {out / 'component_evals.csv'}")
    for key, filename in TABLE_KEYS.items():
        if tables[key]:
            pd.concat(tables[key], ignore_index=True).to_csv(out / filename, index=False)
            print(f"{len(tables[key])} {key} tables -> {out / filename}")


if __name__ == "__main__":
    main()
