"""Cross-checkpoint comparison plots for eval sweep tables."""

import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_MODE_COLORS = {"raw": "tab:blue", "ae": "tab:olive", "vae": "tab:pink"}


_METHOD_COLORS = {"lewm": "tab:blue", "pldm": "tab:red", "smwm": "tab:green"}

_VARIANT_COLORS = ["tab:pink", "tab:cyan", "tab:olive", "tab:gray"]

_PLANNER_COLORS = {
    "CEMSolver": "tab:blue", "ICEMSolver": "tab:orange", "MPPISolver": "tab:purple",
    "PredictiveSamplingSolver": "tab:brown",
}

_MODE_ORDER = {"raw": 0, "ae": 1, "vae": 2}

WHITENED_SUFFIX = " (whitened)"

CHOSEN_TAG = "detach0_dec1_enc1"

ENV_NAMES = {"cube": "Cube", "pusht": "Push-T", "reacher": "Reacher", "tworoom": "Two-Rooms", "antmaze": "AntMaze"}
METHOD_NAMES = {"lewm": "LeWM", "pldm": "PLDM", "smwm": "SMWM"}
ARM_NAMES = {"raw": "Raw actions", "vae": "VAE skills", "ae": "AE skills"}
PLANNER_NAMES = {"CEMSolver": "CEM", "ICEMSolver": "iCEM", "MPPISolver": "MPPI", "PredictiveSamplingSolver": "PS"}
TAG_NAMES = {"pos": "pos. emb.", "posthocdec": "post-hoc decoder"}
SUCCESS_LABEL = "Success rate (%)"
AXIS_NAMES = {
    "goal_offset_steps": "Goal distance (steps)", "goal_cell_distance": "Goal distance (maze cells)",
    "env_steps": "Env steps",
}
_ENV_ORDER = ["cube", "pusht", "reacher", "tworoom", "antmaze"]


def _ordered_envs(df) -> list:
    return sorted(df["env"].unique(), key=lambda env: _ENV_ORDER.index(env) if env in _ENV_ORDER else len(_ENV_ORDER))


def display_label(label: str) -> str:
    """A shortened arm label as the figure shows it: 'lewm/vae + pos (whitened)' -> 'VAE skills + pos. emb. (whitened)'."""
    rest = label.split("/", 1)[-1]
    suffix = ""
    if rest.endswith(WHITENED_SUFFIX):
        rest, suffix = rest.removesuffix(WHITENED_SUFFIX), " (whitened)"
    match = re.fullmatch(r"(raw|ae|vae)(.*)", rest)
    if match is None:
        return label
    mode, tail = match.groups()
    names = [ARM_NAMES[mode]]
    for tag in tail.split(" + "):
        tag = tag.strip()
        if tag:
            names.append(TAG_NAMES.get(tag, tag))
    return " + ".join(names) + suffix


def _method(label: str) -> str | None:
    return label.split("/")[0] if "/" in label else None


def is_multi_method(df) -> bool:
    return df["label"].map(_method).dropna().nunique() > 1


def shorten_labels(labels: pd.Series) -> pd.Series:
    """Drops CHOSEN_TAG when it is the only detach tag present, else just detach0_; _pos and _posthocdec become ' + ' suffixes."""
    tags = labels.str.extract(r"(detach\d_dec\d_enc\d)")[0].dropna().unique()
    if list(tags) != [CHOSEN_TAG]:
        return labels.str.replace(" detach0_", " ", regex=False)
    return (
        labels.str.replace(f" {CHOSEN_TAG}", "", regex=False)
        .str.replace("_posthocdec", " + posthocdec", regex=False)
        .str.replace("_pos", " + pos", regex=False)
    )


def _line_style(label: str) -> str:
    if label.endswith("raw"):
        return "--"
    return ":" if WHITENED_SUFFIX in label else "-"


def build_label_colors(df) -> dict:
    """The method's hue for a plain raw-vs-vae comparison; otherwise one hue per arm, non-raw variants one each when they share an arm."""
    pairs = df[["label", "mode"]].drop_duplicates()
    if is_multi_method(df):
        return {label: _METHOD_COLORS[_method(label)] for label in pairs["label"]}
    methods = pairs["label"].map(_method).dropna().unique()
    if sorted(pairs["mode"]) == ["raw", "vae"] and len(methods) == 1 and methods[0] in _METHOD_COLORS:
        return {label: _METHOD_COLORS[methods[0]] for label in pairs["label"]}
    colors = {label: _MODE_COLORS[mode] for label, mode in pairs.itertuples(index=False)}
    variants = pairs[pairs["mode"] != "raw"]
    tags = variants["label"].str.replace(WHITENED_SUFFIX, "", regex=False)
    if variants.assign(tag=tags).drop_duplicates(["mode", "tag"])["mode"].duplicated().any():
        for i, label in enumerate(sorted(variants["label"])):
            colors[label] = _VARIANT_COLORS[i]
    return colors


def ordered_labels(df) -> list:
    """Legend order: arm, then label; method first when several methods share the plot."""
    key = df[["label", "mode"]].drop_duplicates()
    if is_multi_method(df):
        return sorted(key["label"])
    key["_m"] = key["mode"].map(lambda m: _MODE_ORDER.get(m, len(_MODE_ORDER)))
    return list(key.sort_values(["_m", "label"])["label"])


_PANEL_W, _PANEL_H = 1.35, 1.1
_GAP_W = 0.15
_GAP_H, _GAP_H_TITLED = 0.12, 0.45
_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_BOTTOM = 0.75, 0.12, 0.42
_COLUMN_TITLE_H, _LEGEND_ROW_H = 0.24, 0.17
_TICK_PT, _LABEL_PT = 7, 8


def _legend_entry_width(names: list[str]) -> float:
    """An estimate of the widest legend entry at 8 pt, handle included."""
    return 0.45 + 0.057 * max(len(name) for name in names)


def _legend_ncol(names: list[str], width: float) -> int:
    return max(1, min(len(names), int((width - 0.2) // _legend_entry_width(names))))


def _panel_figure(nrows, ncols, *, sharex, titled_rows: bool, legend_names: list[str]):
    """Fixed-size panels on one y-axis at fixed spacing; only the panel count, legend rows and a too-wide legend entry change the figure size."""
    gap_h = _GAP_H_TITLED if titled_rows else _GAP_H
    panels_right = _MARGIN_LEFT + ncols * _PANEL_W + (ncols - 1) * _GAP_W
    width = panels_right + _MARGIN_RIGHT
    if legend_names:
        width = max(width, _legend_entry_width(legend_names) + 0.2)
    legend_rows = math.ceil(len(legend_names) / _legend_ncol(legend_names, width)) if legend_names else 0
    top = _COLUMN_TITLE_H + legend_rows * _LEGEND_ROW_H
    height = top + nrows * _PANEL_H + (nrows - 1) * gap_h + _MARGIN_BOTTOM
    fig = plt.figure(figsize=(width, height))
    axes = fig.subplots(
        nrows, ncols, sharex=sharex, sharey=True, squeeze=False,
        gridspec_kw=dict(left=_MARGIN_LEFT / width, right=panels_right / width,
                         bottom=_MARGIN_BOTTOM / height, top=1 - top / height,
                         wspace=_GAP_W / _PANEL_W, hspace=gap_h / _PANEL_H),
    )
    for ax in axes.flat:
        ax.tick_params(labelsize=_TICK_PT)
        ax.grid(alpha=0.3)
    return fig, axes


def _finish_panel_figure(fig, axes, *, column_x: list[str], ylabel: str, handles: list, names: list[str], path: Path) -> None:
    """Axis labels in display names (one x label per run of columns sharing an x), the legend on top, saved as PDF and PNG."""
    width, height = fig.get_size_inches()
    bottom_row = axes[-1]
    start = 0
    for end in range(1, len(column_x) + 1):
        if end < len(column_x) and column_x[end] == column_x[start]:
            continue
        left, right = bottom_row[start].get_position().x0, bottom_row[end - 1].get_position().x1
        if column_x[start] is not None:
            fig.text((left + right) / 2, 0.06 / height, AXIS_NAMES.get(column_x[start], column_x[start]),
                     ha="center", va="bottom", fontsize=_LABEL_PT)
        start = end
    panels_top, panels_bottom = axes[0][0].get_position().y1, bottom_row[0].get_position().y0
    fig.text(0.06 / width, (panels_top + panels_bottom) / 2, ylabel,
             rotation=90, ha="left", va="center", fontsize=_LABEL_PT)
    if handles:
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.02 / height),
                   ncol=_legend_ncol(names, width), fontsize=_LABEL_PT, frameon=False, handlelength=3,
                   columnspacing=1.2, borderaxespad=0)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    print(f"{path.with_suffix('.pdf')}: {width:.2f} in wide")
    plt.close(fig)


SUCCESS_YLIM = (0, 110)


def _check_diagonal(df) -> None:
    """Each goal offset must map to one plan horizon, or a panel would mix horizons."""
    if df.groupby("goal_offset_steps")["plan_horizon"].nunique().max() > 1:
        raise ValueError("several plan horizons per goal offset; grid plots assume one")


_SEED_KEYS = [
    "env", "frameskip", "cost_mode", "loop_mode", "label", "mode", "goal_offset_steps", "goal_cell_distance",
    "plan_horizon", "eval_budget", "solver", "cem_num_samples", "cem_topk", "whiten_skill_space",
]


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per arm and cell: success rate's mean and std (success_rate_std) over training seeds."""
    grouped = df.groupby(_SEED_KEYS, dropna=False)["success_rate"]
    return grouped.mean().to_frame().join(grouped.std().rename("success_rate_std")).reset_index()


def _draw(ax, line, x, y, **style) -> None:
    """A line through the seed means, with std error bars where there is more than one seed."""
    std = line[f"{y}_std"] if f"{y}_std" in line else None
    if std is None or std.isna().all():
        ax.plot(line[x], line[y], marker="o", markersize=3, **style)
        return
    ax.errorbar(line[x], line[y], yerr=std.fillna(0), marker="o", markersize=3, capsize=2, **style)


def _x_for(env_df, x) -> str:
    """Cell-distance sweeps plot against maze-cell distance where the other envs plot against goal steps."""
    if x == "goal_offset_steps" and "goal_cell_distance" in env_df and env_df["goal_cell_distance"].notna().all():
        return "goal_cell_distance"
    return x


def _env_grid(df, *, x, colors, path) -> None:
    """One panel per env (2x2 for four envs, else one row), one line per label, one legend on top."""
    envs = _ordered_envs(df)
    nrows, ncols = (2, 2) if len(envs) == 4 else (1, len(envs))
    labels = ordered_labels(df)
    names = [display_label(label) for label in labels]
    fig, axes = _panel_figure(nrows, ncols, sharex=False, titled_rows=True, legend_names=names)
    column_x = [None] * ncols

    for i, env in enumerate(envs):
        ax = axes.flat[i]
        env_df = df[df["env"] == env]
        env_x = _x_for(env_df, x)
        column_x[i % ncols] = env_x
        for label in labels:
            line = env_df[env_df["label"] == label].sort_values(env_x)
            if line.empty:
                continue
            _draw(ax, line, env_x, "success_rate", color=colors[label], linestyle=_line_style(label))
        ax.set_title(ENV_NAMES.get(env, env), fontsize=_LABEL_PT, pad=3)
        ax.set_xticks(sorted(env_df[env_x].unique()))

    axes.flat[0].set_ylim(*SUCCESS_YLIM)
    handles = [plt.Line2D([], [], color=colors[label], linestyle=_line_style(label), marker="o", markersize=3)
               for label in labels]
    _finish_panel_figure(fig, axes, column_x=column_x, ylabel=SUCCESS_LABEL, handles=handles, names=names, path=path)


def _method_grid(df, *, x, path) -> None:
    """One row per method, one column per env (a single row of methods for one env); raw dashed, vae solid, shaded where vae is above raw."""
    methods = sorted(df["label"].map(_method).dropna().unique())
    envs = _ordered_envs(df)
    single_env = len(envs) == 1
    names = [ARM_NAMES["raw"], ARM_NAMES["vae"]]
    nrows, ncols = (1, len(methods)) if single_env else (len(methods), len(envs))
    fig, axes = _panel_figure(nrows, ncols, sharex="col", titled_rows=False, legend_names=names)
    column_x = [_x_for(df[df["env"] == env], x) for env in envs] * (len(methods) if single_env else 1)

    for row, method in enumerate(methods):
        color = _METHOD_COLORS[method]
        for col, env in enumerate(envs):
            ax = axes[0][row] if single_env else axes[row][col]
            env_x = _x_for(df[df["env"] == env], x)
            cell = df[(df["env"] == env) & (df["label"].map(_method) == method)]
            raw_line = cell[cell["mode"] == "raw"].sort_values(env_x)
            vae_line = cell[cell["mode"] == "vae"].sort_values(env_x)
            if not raw_line.empty:
                _draw(ax, raw_line, env_x, "success_rate", color=color, linestyle="--")
            if not vae_line.empty:
                _draw(ax, vae_line, env_x, "success_rate", color=color, linestyle="-")
            _shade_vae_above_raw(ax, raw_line, vae_line, env_x, color)
            ax.set_xticks(sorted(df[df["env"] == env][env_x].unique()))
            if single_env:
                ax.set_title(METHOD_NAMES.get(method, method), fontsize=_LABEL_PT, pad=3)
                continue
            if row == 0:
                ax.set_title(ENV_NAMES.get(env, env), fontsize=_LABEL_PT, pad=3)
            if col == 0:
                ax.set_ylabel(METHOD_NAMES.get(method, method), fontsize=_LABEL_PT)

    axes[0][0].set_ylim(*SUCCESS_YLIM)
    _finish_panel_figure(fig, axes, column_x=column_x, ylabel=SUCCESS_LABEL, handles=_RAW_VAE_HANDLES, names=names, path=path)


_RAW_VAE_HANDLES = [plt.Line2D([], [], color="gray", linestyle="--", marker="o", markersize=3),
                    plt.Line2D([], [], color="gray", linestyle="-", marker="o", markersize=3)]


def _shade_vae_above_raw(ax, raw_line, vae_line, x, color) -> None:
    """Fills between the raw and vae success means only where vae is higher."""
    raw = raw_line.set_index(x)["success_rate"]
    vae = vae_line.set_index(x)["success_rate"]
    common = sorted(raw.index.intersection(vae.index))
    if not common:
        return
    raw, vae = raw[common].to_numpy(), vae[common].to_numpy()
    ax.fill_between(common, raw, vae, where=vae > raw, interpolate=True, color=color, alpha=0.2, linewidth=0)


def is_planner_comparison(df) -> bool:
    return "solver" in df and df["solver"].nunique() > 1


def _planner_grid(df, *, x, path) -> None:
    """One row per planner, one column per env; raw dashed, vae solid, shaded where vae is above raw."""
    solvers = [s for s in _PLANNER_ORDER if (df["solver"] == s).any()]
    envs = _ordered_envs(df)
    names = [ARM_NAMES["raw"], ARM_NAMES["vae"]]
    fig, axes = _panel_figure(len(solvers), len(envs), sharex="col", titled_rows=False, legend_names=names)
    column_x = [_x_for(df[df["env"] == env], x) for env in envs]

    for row, solver in enumerate(solvers):
        color = _PLANNER_COLORS[solver]
        for col, env in enumerate(envs):
            ax = axes[row][col]
            env_x = column_x[col]
            cell = df[(df["env"] == env) & (df["solver"] == solver)]
            raw_line = cell[cell["mode"] == "raw"].sort_values(env_x)
            vae_line = cell[cell["mode"] == "vae"].sort_values(env_x)
            if not raw_line.empty:
                _draw(ax, raw_line, env_x, "success_rate", color=color, linestyle="--")
            if not vae_line.empty:
                _draw(ax, vae_line, env_x, "success_rate", color=color, linestyle="-")
            _shade_vae_above_raw(ax, raw_line, vae_line, env_x, color)
            if row == 0:
                ax.set_title(ENV_NAMES.get(env, env), fontsize=_LABEL_PT, pad=3)
            if col == 0:
                ax.set_ylabel(_planner_name(solver), fontsize=_LABEL_PT)
            ax.set_xticks(sorted(df[df["env"] == env][env_x].unique()))

    axes[0][0].set_ylim(*SUCCESS_YLIM)
    _finish_panel_figure(fig, axes, column_x=column_x, ylabel=SUCCESS_LABEL, handles=_RAW_VAE_HANDLES, names=names, path=path)


def is_whitening_comparison(df) -> bool:
    return bool((df["mode"] == "ae").any() and df["whiten_skill_space"].any())


def _whitening_grid(df, *, x, path) -> None:
    """One panel per env (2x2 for four envs, else one row); ae and vae in their colors, plain solid, whitened dotted; raw dashed."""
    modes = [m for m in ("ae", "vae") if (df["mode"] == m).any()]
    envs = _ordered_envs(df)
    nrows, ncols = (2, 2) if len(envs) == 4 else (1, len(envs))
    handles, names = [], []
    if (df["mode"] == "raw").any():
        handles.append(plt.Line2D([], [], color=_MODE_COLORS["raw"], linestyle="--", marker="o", markersize=3))
        names.append(ARM_NAMES["raw"])
    for mode in modes:
        for suffix, ls in (("", "-"), (" (whitened)", ":")):
            handles.append(plt.Line2D([], [], color=_MODE_COLORS[mode], linestyle=ls, marker="o", markersize=3))
            names.append(f"{ARM_NAMES[mode]}{suffix}")
    fig, axes = _panel_figure(nrows, ncols, sharex=False, titled_rows=True, legend_names=names)

    for i, env in enumerate(envs):
        ax = axes.flat[i]
        env_df = df[df["env"] == env]
        _draw(ax, env_df[env_df["mode"] == "raw"].sort_values(x), x, "success_rate", color=_MODE_COLORS["raw"], linestyle="--")
        for mode in modes:
            cell = env_df[env_df["mode"] == mode]
            _draw(ax, cell[~cell["whiten_skill_space"]].sort_values(x), x, "success_rate", color=_MODE_COLORS[mode], linestyle="-")
            _draw(ax, cell[cell["whiten_skill_space"]].sort_values(x), x, "success_rate", color=_MODE_COLORS[mode], linestyle=":")
        ax.set_title(ENV_NAMES.get(env, env), fontsize=_LABEL_PT, pad=3)
        ax.set_xticks(sorted(env_df[x].unique()))

    axes.flat[0].set_ylim(*SUCCESS_YLIM)
    _finish_panel_figure(fig, axes, column_x=[x] * ncols, ylabel=SUCCESS_LABEL, handles=handles, names=names, path=path)


def _latex_cell(mean: float, std: float, fmt: str) -> str:
    if np.isnan(mean):
        return "--"
    if np.isnan(std):
        return fmt.format(mean)
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"


def _latex_text(text: str) -> str:
    return text.replace("_", r"\_")


def _table_column(label: str, solver: str | None) -> tuple[str, str]:
    """(group, arm) header for a label: the planner when solver is given, else the method prefix."""
    if solver is not None:
        arm = label.removesuffix(f" ({solver})")
        arm = arm.split("/", 1)[1] if "/" in arm else arm
        return _planner_name(solver), arm
    if "/" in label:
        group, arm = label.split("/", 1)
        return group, arm
    return "", label


def _latex_table(df, *, rows, caption, label, path) -> None:
    """Rows grouped by method or planner with its arms beneath, one column per env holding its goal offsets joined by /; the higher arm per group bolded."""
    solver_of = dict(zip(df["label"], df["solver"])) if is_planner_comparison(df) else {}
    columns = {row: _table_column(row, solver_of.get(row)) for row in rows}
    if solver_of:
        planner_rank = {_planner_name(s): i for i, s in enumerate(_PLANNER_ORDER)}
        rows = sorted(rows, key=lambda row: planner_rank.get(columns[row][0], len(planner_rank)))
    mode_of = dict(zip(df["label"], df["mode"]))
    groups = {}
    for row in rows:
        groups.setdefault(columns[row][0], []).append(row)
    for members in groups.values():
        members.sort(key=lambda row: _MODE_ORDER.get(mode_of[row], len(_MODE_ORDER)))
    rows = [row for members in groups.values() for row in members]

    columns_by = ["env", "goal_offset_steps"]
    mean = df.pivot_table(index="label", columns=columns_by, values="success_rate", dropna=False).reindex(rows)
    std = df.pivot_table(index="label", columns=columns_by, values="success_rate_std", dropna=False).reindex(index=rows, columns=mean.columns)

    envs = list(dict.fromkeys(mean.columns.get_level_values(0)))
    cells = []
    for row in rows:
        members = groups[columns[row][0]]
        line = []
        for env in envs:
            parts = []
            for col in mean[env].columns:
                text = _latex_cell(mean.loc[row, (env, col)], std.loc[row, (env, col)], "{:.0f}").replace(" $\\pm$ ", "$\\pm$")
                if len(members) > 1 and mean.loc[row, (env, col)] == mean.loc[members, (env, col)].max():
                    text = f"\\textbf{{{text}}}"
                parts.append(text)
            line.append("\\,/\\,".join(parts))
        cells.append(line)

    table = pd.DataFrame(
        cells,
        index=pd.MultiIndex.from_tuples([(_latex_text(g), _latex_text(a)) for g, a in (columns[row] for row in rows)]),
        columns=[f"{env} ({'\\,/\\,'.join(f'{v:g}' for v in mean[env].columns)})" for env in envs],
    )
    latex = table.to_latex(
        multicolumn_format="c", escape=False, column_format="ll" + "c" * len(table.columns),
        caption=_latex_text(caption), label=label, position="t",
    )
    latex = latex.replace("\\begin{table}", "\\begin{table*}", 1).replace("\\end{table}", "\\end{table*}", 1)
    path.write_text(latex.replace(
        "\\begin{tabular}", "\\centering\n\\footnotesize\n\\renewcommand{\\arraystretch}{1.3}\n\\begin{tabular}", 1
    ))


def _success_plots(df, figures_dir: Path, tables_dir: Path, label_prefix: str, colors) -> None:
    """Success rate vs. goal distance as one table and one grid: a planner, whitening, method or env grid."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _check_diagonal(df)
    df = df[df["success_rate"].notna()]
    cell_distance_sweep = df["goal_cell_distance"].notna().all()
    x = "goal_cell_distance" if cell_distance_sweep else "goal_offset_steps"
    path = figures_dir / "success_rate_vs_distance.png"
    if df["env"].nunique() == 1:
        success = df.pivot_table(index="label", columns="goal_offset_steps", values="success_rate")
        rows = list(success.mean(axis=1).sort_values(ascending=False).index)
    else:
        rows = ordered_labels(df)
    if not cell_distance_sweep:
        _latex_table(df, rows=rows, caption="Success rate vs. goal distance.",
                     label=f"{label_prefix}:success_rate_vs_distance", path=tables_dir / "success_rate_vs_distance.tex")
    if is_planner_comparison(df):
        _planner_grid(df, x=x, path=path)
    elif is_whitening_comparison(df):
        _whitening_grid(df, x=x, path=path)
    elif is_multi_method(df):
        _method_grid(df, x=x, path=path)
    else:
        _env_grid(df, x=x, colors=colors, path=path)
    print(f"Saved plots to {figures_dir}/ and {tables_dir}/")


def sweep_parts(output_dir: Path) -> list[str]:
    return [output_dir.resolve().name]


_PLANNER_ORDER = ["CEMSolver", "ICEMSolver", "MPPISolver", "PredictiveSamplingSolver"]


def _planner_name(solver: str) -> str:
    return PLANNER_NAMES.get(solver, solver.removesuffix("Solver"))


def paired_vae_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (env, solver, cell, seed) with the raw and vae success rates and delta = vae - raw."""
    arms = df[df["mode"].isin(["raw", "vae"])]
    if arms.groupby(["env", "solver", "mode"])["label"].nunique().max() > 1:
        raise ValueError("several raw or vae arms per (env, solver); the delta needs exactly one of each")
    keys = ["env", "solver", "cost_mode", "loop_mode", "goal_offset_steps", "goal_cell_distance", "seed"]
    keys = [k for k in keys if k in arms and arms[k].notna().all()]
    wide = arms.pivot_table(index=keys, columns="mode", values="success_rate").dropna(subset=["raw", "vae"])
    return wide.assign(delta=wide["vae"] - wide["raw"]).reset_index()


def _planner_delta_bars(deltas: pd.DataFrame, path: Path) -> None:
    """One panel per env (2x2 for four envs, else one row), one bar per planner: vae - raw averaged over goal distances per seed, mean ± std over seeds."""
    per_seed = deltas.groupby(["env", "solver", "seed"])["delta"].mean()
    summary = per_seed.groupby(["env", "solver"]).agg(["mean", "std"])
    solvers = [s for s in _PLANNER_ORDER if s in summary.index.get_level_values("solver")]
    envs = _ordered_envs(deltas)
    nrows, ncols = (2, 2) if len(envs) == 4 else (1, len(envs))
    fig, axes = _panel_figure(nrows, ncols, sharex=False, titled_rows=True, legend_names=[])
    positions = np.arange(len(solvers))

    for ax, env in zip(axes.flat, envs):
        means = [summary.loc[(env, s), "mean"] for s in solvers]
        stds = [summary.loc[(env, s), "std"] for s in solvers]
        ax.bar(positions, means, width=0.7, color=[_PLANNER_COLORS[s] for s in solvers],
               yerr=stds, capsize=2, error_kw=dict(elinewidth=0.8, capthick=0.8))
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(positions, [_planner_name(s) for s in solvers])
        ax.grid(False, axis="x")
        ax.set_title(ENV_NAMES.get(env, env), fontsize=_LABEL_PT, pad=3)

    _finish_panel_figure(fig, axes, column_x=[None] * ncols, ylabel="Δ Success rate (pp)", handles=[], names=[], path=path)


def make_planner_delta_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """The vae - raw success bars per planner, into the figures_ dir make_plots wrote."""
    if df["frameskip"].nunique() > 1 or len(df[["cost_mode", "loop_mode"]].drop_duplicates()) > 1:
        raise ValueError("planner delta plots assume one frameskip and one condition")
    figures_dir = output_dir / f"figures_{'_'.join(sweep_parts(output_dir))}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _planner_delta_bars(paired_vae_deltas(df), figures_dir / "planner_delta_success_rate_vs_distance_bars.png")
    print(f"Saved planner delta plot to {figures_dir}/")


def make_iteration0_kl_table(kl: pd.DataFrame, output_dir: Path) -> None:
    """KL(fitted skill prior || N(0, I)), mean ± std over seeds: one row per skill arm, one column per env."""
    parts = sweep_parts(output_dir)
    tables_dir = output_dir / f"tables_{'_'.join(parts)}"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = kl.groupby(["env", "mode"])["kl"].agg(["mean", "std"])
    envs = _ordered_envs(kl)
    rows = [("ae", "AE", 1.0, "{:.2f}"),
            ("vae", "VAE ($\\times 10^{-4}$)", 1e4, "{:.1f}")]

    lines = []
    for mode, name, scale, fmt in rows:
        cells = [f"${fmt.format(summary.loc[(env, mode), 'mean'] * scale)} \\pm "
                 f"{fmt.format(summary.loc[(env, mode), 'std'] * scale)}$"
                 for env in envs]
        lines.append(" & ".join([name, *cells]) + " \\\\")

    n_seeds = kl["seed"].nunique()
    caption = (
        r"$\mathrm{KL}(\hat q \,\|\, \mathcal{N}(0, I))$ per plan step, in nats, where $\hat q$ is a diagonal Gaussian "
        r"fitted to the skill encoder's outputs on the training set (for VAE skills, sampled latents) and "
        r"$\mathcal{N}(0, I)$ is CEM's initial sampling distribution. "
        rf"Mean $\pm$ std over {n_seeds} training seeds."
    )
    latex = "\n".join([
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        "\\label{tab:" + ":".join(parts) + ":iteration0_kl}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}l" + "c" * len(envs) + "@{}}",
        "\\toprule",
        " & ".join(["", *[ENV_NAMES.get(env, env) for env in envs]]) + " \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = tables_dir / "iteration0_kl.tex"
    path.write_text(latex)
    print(f"Saved iteration-0 KL table to {path}")


def _execution_grid(control: pd.DataFrame, *, frameskip: int, path: Path, ylim: tuple[float, float] | None = None) -> None:
    """The model's rollout vs the executed frames per step over persistence, one panel per env on one y-axis."""
    control = control.groupby(["env", "label", "mode", "step"], as_index=False)["plan_execution_over_copy"].agg(
        plan_execution_over_copy="mean", plan_execution_over_copy_std="std"
    )
    control["env_steps"] = control["step"] * frameskip
    envs = _ordered_envs(control)
    labels = ordered_labels(control)
    names = [display_label(label) for label in labels]
    colors = {label: _MODE_COLORS[mode] for label, mode in control[["label", "mode"]].drop_duplicates().itertuples(index=False)}
    fig, axes = _panel_figure(2, 2, sharex=False, titled_rows=True, legend_names=names)

    for ax, env in zip(axes.flat, envs):
        for label in labels:
            line = control[(control["env"] == env) & (control["label"] == label)].sort_values("step")
            _draw(ax, line, "env_steps", "plan_execution_over_copy", color=colors[label], linestyle=_line_style(label))
        ax.set_title(ENV_NAMES.get(env, env), fontsize=_LABEL_PT, pad=3)
        ax.set_xticks([20, 40, 60, 80])
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    handles = [plt.Line2D([], [], color=colors[label], linestyle=_line_style(label), marker="o", markersize=3) for label in labels]
    _finish_panel_figure(fig, axes, column_x=["env_steps", "env_steps"], ylabel="Predicted vs. executed error / persistence",
                         handles=handles, names=names, path=path)


def make_execution_plots(dataset: pd.DataFrame, iteration0: pd.DataFrame, output_dir: Path) -> None:
    """Predicted-vs-executed error per step for every arm, on dataset plans and on CEM first-iteration candidates."""
    figures_dir = output_dir / f"figures_{'_'.join(sweep_parts(output_dir))}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _execution_grid(dataset.assign(label=shorten_labels(dataset["label"])), frameskip=int(dataset["frameskip"].iloc[0]),
                    path=figures_dir / "control_plan_execution_by_step.png")
    _execution_grid(iteration0.assign(label=shorten_labels(iteration0["label"])), frameskip=int(iteration0["frameskip"].iloc[0]),
                    path=figures_dir / "iteration0_plan_execution_by_step.png", ylim=(0, 1))
    print(f"Saved plan-execution plots to {figures_dir}/")


def _mean_success_table(per_seed: pd.DataFrame, path: Path) -> None:
    """Success rate per arm and goal distance: mean over seeds per (task, world model), then mean over those pairs."""
    df = per_seed.assign(method=per_seed["label"].map(_method))
    per_pair = df.groupby(["env", "method", "mode", "goal_offset_steps"], as_index=False)["success_rate"].mean()
    n_pairs = per_pair.groupby(["mode", "goal_offset_steps"])["success_rate"].size()
    mean = per_pair.pivot_table(index="mode", columns="goal_offset_steps", values="success_rate", aggfunc="mean")
    modes = [mode for mode in ("raw", "vae") if mode in mean.index]
    print(f"Mean success rate over {n_pairs.min()}-{n_pairs.max()} task x world-model pairs:")
    print(mean.loc[modes].to_string())

    table = pd.DataFrame(
        [[f"{mean.loc[mode, offset]:.1f}" for offset in mean.columns] for mode in modes],
        index=[ARM_NAMES[mode] for mode in modes],
        columns=[f"{offset:g}" for offset in mean.columns],
    )
    latex = table.to_latex(
        escape=False, column_format="l" + "c" * len(table.columns), position="t",
        caption="Mean success rate (\\%) vs. goal distance, averaged over the four tasks and three world models.",
        label="tab:mean_success_vs_distance",
    )
    path.write_text(latex.replace(
        "\\begin{tabular}", "\\centering\n\\footnotesize\n\\renewcommand{\\arraystretch}{1.3}\n\\begin{tabular}", 1
    ))


def _antmaze_tables(per_seed: pd.DataFrame, tables_dir: Path) -> None:
    """AntMaze success per world model and arm (mean ± std over seeds), and its mean over world models, per maze-cell distance."""
    df = per_seed.assign(method=per_seed["label"].map(_method))
    per_wm = df.groupby(["method", "mode", "goal_cell_distance"])["success_rate"].agg(["mean", "std"])
    distances = sorted(df["goal_cell_distance"].unique())
    methods = [m for m in METHOD_NAMES if m in per_wm.index.get_level_values("method")]
    modes = ["raw", "vae"]

    lines = []
    for i, method in enumerate(methods):
        if i:
            lines.append("\\midrule")
        for j, mode in enumerate(modes):
            cells = []
            for d in distances:
                mean, std = per_wm.loc[(method, mode, d), "mean"], per_wm.loc[(method, mode, d), "std"]
                text = _latex_cell(mean, std, "{:.0f}").replace(" $\\pm$ ", "$\\pm$")
                if mean == max(per_wm.loc[(method, m, d), "mean"] for m in modes):
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            group = f"\\multirow[t]{{2}}{{*}}{{{METHOD_NAMES[method]}}}" if j == 0 else ""
            lines.append(" & ".join([group, ARM_NAMES[mode], *cells]) + " \\\\")
    header = " & ".join(["", "", *[f"{d:g}" for d in distances]]) + " \\\\"
    (tables_dir / "antmaze_success.tex").write_text("\n".join([
        "\\begin{table}[t]",
        "\\caption{Success rate (\\%) vs. goal distance (maze cells) on AntMaze. Mean $\\pm$ std over 3 training seeds.}",
        "\\label{tab:antmaze:success_rate_vs_distance}",
        "\\centering",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.3}",
        "\\begin{tabular}{ll" + "c" * len(distances) + "}",
        "\\toprule",
        header,
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]))

    mean = per_wm["mean"].groupby(["mode", "goal_cell_distance"]).mean().unstack("goal_cell_distance")
    table = pd.DataFrame(
        [[f"{mean.loc[mode, d]:.1f}" for d in distances] for mode in modes],
        index=[ARM_NAMES[mode] for mode in modes],
        columns=[f"{d:g}" for d in distances],
    )
    latex = table.to_latex(
        escape=False, column_format="l" + "c" * len(distances), position="t",
        caption="Mean success rate (\\%) vs. goal distance on AntMaze, averaged over the three world models of "
                "Table~\\ref{tab:antmaze:success_rate_vs_distance}.",
        label="tab:antmaze:mean_success_vs_distance",
    )
    (tables_dir / "antmaze_mean_success.tex").write_text(latex.replace(
        "\\begin{tabular}", "\\centering\n\\footnotesize\n\\renewcommand{\\arraystretch}{1.3}\n\\begin{tabular}", 1
    ))


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """The success-rate figure and tables into figures_<name>/ and tables_<name>/, <name> being the output directory's name."""
    if df["frameskip"].nunique() > 1 or len(df[["cost_mode", "loop_mode"]].drop_duplicates()) > 1:
        raise ValueError("plots assume one frameskip and one cost/loop condition per pull")
    for old in [*output_dir.glob("figures_*"), *output_dir.glob("tables_*")]:
        shutil.rmtree(old)
    parts = sweep_parts(output_dir)
    name = "_".join(parts)
    tables_dir = output_dir / f"tables_{name}"
    seeds = aggregate_seeds(df.assign(label=shorten_labels(df["label"])))
    _success_plots(seeds, output_dir / f"figures_{name}", tables_dir, "tab:" + ":".join(parts), build_label_colors(seeds))
    if is_multi_method(seeds) and seeds["env"].nunique() > 1:
        _mean_success_table(df, tables_dir / "mean_success_vs_distance.tex")
    if is_multi_method(seeds) and seeds["goal_cell_distance"].notna().all():
        _antmaze_tables(df, tables_dir)
