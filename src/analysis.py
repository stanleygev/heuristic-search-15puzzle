"""
analysis.py - Load benchmark results and produce figures + summary tables.

Outputs (saved to results/figures/):
  fig1_nodes_comparison.png       Bar chart, nodes expanded by algorithm
  fig2_time_comparison.png        Bar chart, wall-clock time by algorithm
  fig3_weighted_tradeoff.png      Line chart, solution quality vs runtime
  fig4_difficulty_scaling.png     Line chart, runtime vs difficulty tier
  fig5_heuristic_quality.png      Box plot, h(start) vs optimal solution length
  fig6_memory_comparison.png      Bar chart, peak memory usage by algorithm

All figures use matplotlib with publication-quality defaults.
"""

from __future__ import annotations
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS = {
    "A*+Manhattan":         "#2196F3",
    "A*+LinearConflict":    "#4CAF50",
    "A*+PDB(5-5-5)":        "#FF9800",
    "A*+PDB(6-5-4)":        "#E65100",
    "A*+PDB(7-8)":          "#F44336",
    "IDA*+Manhattan":       "#03A9F4",
    "IDA*+LinearConflict":  "#8BC34A",
    "IDA*+PDB(5-5-5)":      "#9C27B0",
    "IDA*+PDB(6-5-4)":      "#7B1FA2",
    "WeightedA*(w=1.25)":   "#009688",
    "WeightedA*(w=1.5)":    "#795548",
    "WeightedA*(w=2.0)":    "#607D8B",
    "WIDA*(w=1.5)":         "#5D4037",
}
DEFAULT_COLOR = "#90A4AE"


def _color(name: str) -> str:
    return COLORS.get(name, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return dict(out)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def solved_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["solved"]]


def conf_interval_95(values: list[float]) -> float:
    """Return the half-width of a 95% normal confidence interval on the
    mean (1.96 * sigma / sqrt(n))."""
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / (len(values) ** 0.5)


# ---------------------------------------------------------------------------
# Figure 1 & 2 – Algorithm comparison bar charts
# ---------------------------------------------------------------------------

def plot_algorithm_comparison(
    results: list[dict],
    configs: list[str],
    output_dir: Path,
    common_only: bool = True,
) -> None:
    """Bar charts comparing nodes expanded and runtime across algorithms.

    Error bars show 95% confidence intervals on the mean.

    When *common_only* is True (default), the comparison is restricted
    to instances that were solved by *every* listed config. This prevents
    a weak heuristic that solved only the easiest 4 instances from
    appearing artificially fast.
    """
    by_config = group_by(results, "config")

    # Determine the common set of solved instance indices, if requested.
    common_idx: set[int] | None = None
    if common_only:
        for cfg in configs:
            solved = {r["instance_idx"] for r in solved_rows(by_config.get(cfg, []))}
            common_idx = solved if common_idx is None else common_idx & solved
        if not common_idx:
            common_idx = None  # fall back to any-solved if intersection empty

    nodes_mean: list[float] = []
    nodes_ci: list[float] = []
    time_mean: list[float] = []
    time_ci: list[float] = []
    labels: list[str] = []

    for cfg in configs:
        rows = solved_rows(by_config.get(cfg, []))
        if common_idx is not None:
            rows = [r for r in rows if r["instance_idx"] in common_idx]
        if not rows:
            continue
        labels.append(cfg)
        n_vals = [r["nodes_expanded"] for r in rows]
        t_vals = [r["elapsed_seconds"] for r in rows]
        nodes_mean.append(statistics.mean(n_vals))
        nodes_ci.append(conf_interval_95(n_vals))
        time_mean.append(statistics.mean(t_vals))
        time_ci.append(conf_interval_95(t_vals))

    if not labels:
        print("  (No solved data to plot)")
        return

    n_common = len(common_idx) if common_idx is not None else None
    suffix = f" (n={n_common} instances solved by all)" if n_common else ""

    x = np.arange(len(labels))
    width = 0.6

    # --- Figure 1: Nodes expanded ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, nodes_mean, width, yerr=nodes_ci, capsize=4,
           color=[_color(l) for l in labels], alpha=0.88, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Nodes Expanded (log scale)", fontsize=11)
    ax.set_title("Algorithm Comparison: Mean Nodes Expanded (95% CI)" + suffix, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_nodes_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_nodes_comparison.png")

    # --- Figure 2: Wall-clock time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, time_mean, width, yerr=time_ci, capsize=4,
           color=[_color(l) for l in labels], alpha=0.88, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Time (seconds)", fontsize=11)
    ax.set_title("Algorithm Comparison: Mean Wall-Clock Time (95% CI)" + suffix, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_time_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_time_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3 – Weighted A* trade-off curve
# ---------------------------------------------------------------------------

def plot_weighted_tradeoff(
    results: list[dict],
    baseline_config: str,
    weighted_configs: list[tuple[str, float]],
    output_dir: Path,
) -> None:
    """Line chart: solution quality ratio vs mean runtime for weighted A*."""

    by_config = group_by(results, "config")

    # Optimal lengths from baseline (weight=1.0)
    baseline_rows = solved_rows(by_config.get(baseline_config, []))
    opt_map = {r["instance_idx"]: r["solution_length"] for r in baseline_rows}

    weights, quality_ratios, runtimes = [], [], []

    all_configs = [(baseline_config, 1.0)] + list(weighted_configs)

    for cfg_name, w in all_configs:
        rows = solved_rows(by_config.get(cfg_name, []))
        rows_with_opt = [r for r in rows if r["instance_idx"] in opt_map]
        if not rows_with_opt:
            continue
        ratios = [r["solution_length"] / opt_map[r["instance_idx"]] for r in rows_with_opt]
        times = [r["elapsed_seconds"] for r in rows_with_opt]
        weights.append(w)
        quality_ratios.append(statistics.mean(ratios))
        runtimes.append(statistics.mean(times))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_r = "#F44336"
    color_t = "#2196F3"

    ax1.plot(weights, quality_ratios, "o-", color=color_r, linewidth=2,
             markersize=8, label="Mean solution length / optimal")
    ax1.set_xlabel("Weight w", fontsize=11)
    ax1.set_ylabel("Mean Solution Length / Optimal", color=color_r, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_r)
    ax1.set_ylim(bottom=min(0.99, min(quality_ratios) - 0.02))

    ax2 = ax1.twinx()
    ax2.plot(weights, runtimes, "s--", color=color_t, linewidth=2,
             markersize=8, label="Mean runtime (s)")
    ax2.set_ylabel("Mean Runtime (seconds)", color=color_t, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_t)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=9)
    ax1.set_title("Weighted A* Trade-Off: Solution Quality vs Runtime", fontsize=12)
    ax1.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_weighted_tradeoff.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_weighted_tradeoff.png")


# ---------------------------------------------------------------------------
# Figure 4 – Performance by difficulty tier
# ---------------------------------------------------------------------------

def plot_difficulty_scaling(
    results: list[dict],
    configs: list[str],
    output_dir: Path,
) -> None:
    """Line chart: mean runtime per difficulty tier per algorithm."""

    difficulty_order = ["easy", "medium", "hard"]
    by_config = group_by(results, "config")

    fig, ax = plt.subplots(figsize=(9, 5))

    for cfg in configs:
        rows = solved_rows(by_config.get(cfg, []))
        by_diff = group_by(rows, "difficulty")
        means: list[float | None] = []
        for d in difficulty_order:
            d_rows = by_diff.get(d, [])
            if d_rows:
                means.append(statistics.mean([r["elapsed_seconds"] for r in d_rows]))
            else:
                means.append(None)

        valid_x = [i for i, v in enumerate(means) if v is not None]
        valid_y = [v for v in means if v is not None]
        if valid_x:
            ax.plot(valid_x, valid_y, "o-", label=cfg, color=_color(cfg),
                    linewidth=2, markersize=7)

    ax.set_xticks(range(len(difficulty_order)))
    ax.set_xticklabels(difficulty_order, fontsize=11)
    ax.set_ylabel("Mean Runtime (seconds)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Runtime vs. Difficulty Tier (log scale)", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_difficulty_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig4_difficulty_scaling.png")


# ---------------------------------------------------------------------------
# Figure 5 – Heuristic quality (h vs h*)
# ---------------------------------------------------------------------------

def plot_heuristic_quality(
    results: list[dict],
    optimal_config: str,
    output_dir: Path,
) -> None:
    """Box plot: h(start)/h*(start) for each heuristic, where h* is the
    optimal solution length found by *optimal_config*. Closer to 1.0 is
    a more informed (i.e. better) admissible heuristic.
    """
    from puzzle import manhattan_distance, linear_conflict
    by_config = group_by(results, "config")
    opt_rows = solved_rows(by_config.get(optimal_config, []))
    opt_map = {r["instance_idx"]: r["solution_length"] for r in opt_rows}

    # Recompute h(start) values. Only Manhattan and LinearConflict are
    # reliably recomputable without the PDB; PDBs would need to be
    # re-instantiated. We capture what's accessible.
    benchmark_path = Path(__file__).resolve().parent.parent / "data" / "benchmark.json"
    if not benchmark_path.exists():
        return
    with open(benchmark_path) as f:
        instances = [tuple(s) for s in json.load(f)]

    heuristics = {
        "Manhattan": manhattan_distance,
        "LinearConflict": linear_conflict,
    }
    ratios: dict[str, list[float]] = {name: [] for name in heuristics}
    for idx, state in enumerate(instances):
        if idx not in opt_map or opt_map[idx] <= 0:
            continue
        opt = opt_map[idx]
        for name, h in heuristics.items():
            ratios[name].append(h(state) / opt)

    if not any(ratios.values()):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [ratios[k] for k in heuristics if ratios[k]]
    labels = [k for k in heuristics if ratios[k]]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    palette = ["#2196F3", "#4CAF50"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6,
               label="Perfect heuristic (h = h*)")
    ax.set_ylabel("h(start) / optimal length", fontsize=11)
    ax.set_title("Heuristic Informativeness (closer to 1.0 = better)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_heuristic_quality.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig5_heuristic_quality.png")


# ---------------------------------------------------------------------------
# Figure 6 - Memory usage
# ---------------------------------------------------------------------------

def plot_memory_comparison(
    results: list[dict],
    configs: list[str],
    output_dir: Path,
) -> None:
    """Bar chart of peak memory usage by algorithm."""
    by_config = group_by(results, "config")
    means: list[float] = []
    cis: list[float] = []
    labels: list[str] = []
    for cfg in configs:
        rows = solved_rows(by_config.get(cfg, []))
        if not rows:
            continue
        labels.append(cfg)
        vals = [r["peak_memory_mb"] for r in rows]
        means.append(statistics.mean(vals))
        cis.append(conf_interval_95(vals))

    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, 0.6, yerr=cis, capsize=4,
           color=[_color(l) for l in labels], alpha=0.88, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Peak Memory (MB, log scale)", fontsize=11)
    ax.set_title("Algorithm Comparison: Mean Peak Memory (95% CI)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_memory_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig6_memory_comparison.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: list[dict], configs: Iterable[str]) -> None:
    by_config = group_by(results, "config")
    header = (
        f"{'Config':<28} {'Solved':>8} {'MeanLen':>8} "
        f"{'MeanNodes':>14} {'MeanTime':>10} {'PeakMB':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for cfg in configs:
        rows = by_config.get(cfg, [])
        s_rows = solved_rows(rows)
        if not s_rows:
            print(f"{cfg:<28} {'0/' + str(len(rows)):>8}")
            continue
        ml = statistics.mean([r["solution_length"] for r in s_rows])
        mn = statistics.mean([r["nodes_expanded"] for r in s_rows])
        mt = statistics.mean([r["elapsed_seconds"] for r in s_rows])
        mp = statistics.mean([r["peak_memory_mb"] for r in s_rows])
        solve_str = f"{len(s_rows)}/{len(rows)}"
        print(
            f"{cfg:<28} {solve_str:>8} {ml:>8.1f} {mn:>14,.0f} "
            f"{mt:>10.3f} {mp:>10.2f}"
        )


def write_summary_csv(
    results: list[dict],
    configs: Iterable[str],
    out_path: Path,
) -> None:
    """Write a CSV summary suitable for inclusion in the report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_config = group_by(results, "config")
    with open(out_path, "w") as f:
        f.write("config,solved,total,mean_len,mean_nodes,mean_time_s,mean_peak_mb\n")
        for cfg in configs:
            rows = by_config.get(cfg, [])
            s_rows = solved_rows(rows)
            if not s_rows:
                f.write(f"{cfg},0,{len(rows)},,,,\n")
                continue
            ml = statistics.mean([r["solution_length"] for r in s_rows])
            mn = statistics.mean([r["nodes_expanded"] for r in s_rows])
            mt = statistics.mean([r["elapsed_seconds"] for r in s_rows])
            mp = statistics.mean([r["peak_memory_mb"] for r in s_rows])
            f.write(
                f"{cfg},{len(s_rows)},{len(rows)},"
                f"{ml:.2f},{mn:.1f},{mt:.4f},{mp:.2f}\n"
            )
