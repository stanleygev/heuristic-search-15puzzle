"""
analysis.py - Load benchmark results and produce figures / summary tables.

Produces the following outputs (saved to results/figures/):
  fig1_nodes_comparison.png     - Bar chart: nodes expanded by algorithm
  fig2_time_comparison.png      - Bar chart: wall-clock time by algorithm
  fig3_weighted_tradeoff.png    - Line chart: solution quality vs runtime for weighted A*
  fig4_difficulty_scaling.png   - Line chart: runtime vs difficulty tier

All figures use matplotlib with publication-quality defaults.
"""

from __future__ import annotations
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "A*+Manhattan":       "#2196F3",
    "A*+LinearConflict":  "#4CAF50",
    "A*+PDB(5-5-5)":      "#FF9800",
    "A*+PDB(7-8)":        "#F44336",
    "IDA*+PDB(7-8)":      "#9C27B0",
    "WeightedA*(w=1.25)": "#009688",
    "WeightedA*(w=1.5)":  "#795548",
    "WeightedA*(w=2.0)":  "#607D8B",
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


# ---------------------------------------------------------------------------
# Figure 1 & 2 – Algorithm comparison bar charts
# ---------------------------------------------------------------------------

def plot_algorithm_comparison(
    results: list[dict],
    configs: list[str],
    output_dir: Path,
) -> None:
    """Bar charts comparing nodes expanded and runtime across algorithms."""

    by_config = group_by(results, "config")

    # Collect metrics for solved instances only
    nodes_mean, nodes_std = [], []
    time_mean, time_std = [], []
    labels = []

    for cfg in configs:
        rows = solved_rows(by_config.get(cfg, []))
        if not rows:
            continue
        labels.append(cfg)
        n_vals = [r["nodes_expanded"] for r in rows]
        t_vals = [r["elapsed_seconds"] for r in rows]
        nm, ns = mean_std(n_vals)
        tm, ts = mean_std(t_vals)
        nodes_mean.append(nm)
        nodes_std.append(ns)
        time_mean.append(tm)
        time_std.append(ts)

    x = np.arange(len(labels))
    width = 0.6

    # --- Figure 1: Nodes expanded ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, nodes_mean, width, yerr=nodes_std, capsize=4,
                  color=[_color(l) for l in labels], alpha=0.88, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Nodes Expanded (log scale)", fontsize=11)
    ax.set_title("Algorithm Comparison: Mean Nodes Expanded (solved instances)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_nodes_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_nodes_comparison.png")

    # --- Figure 2: Wall-clock time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, time_mean, width, yerr=time_std, capsize=4,
           color=[_color(l) for l in labels], alpha=0.88, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Time (seconds)", fontsize=11)
    ax.set_title("Algorithm Comparison: Mean Wall-Clock Time (solved instances)", fontsize=12)
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

    weights = []
    quality_ratios = []
    runtimes = []

    # Include weight=1.0 (baseline itself)
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

    # Two-axis plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_r = "#F44336"
    color_t = "#2196F3"

    ax1.plot(weights, quality_ratios, "o-", color=color_r, linewidth=2, markersize=7,
             label="Mean Solution Ratio (lower = better quality)")
    ax1.set_xlabel("Weight w", fontsize=11)
    ax1.set_ylabel("Mean Solution Length / Optimal", color=color_r, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_r)
    ax1.set_ylim(bottom=0.95)

    ax2 = ax1.twinx()
    ax2.plot(weights, runtimes, "s--", color=color_t, linewidth=2, markersize=7,
             label="Mean Runtime (s)")
    ax2.set_ylabel("Mean Runtime (seconds)", color=color_t, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_t)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
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
        means = []
        for d in difficulty_order:
            d_rows = by_diff.get(d, [])
            means.append(statistics.mean([r["elapsed_seconds"] for r in d_rows]) if d_rows else None)

        valid_x = [i for i, v in enumerate(means) if v is not None]
        valid_y = [v for v in means if v is not None]
        if valid_x:
            ax.plot(valid_x, valid_y, "o-", label=cfg, color=_color(cfg), linewidth=2, markersize=7)

    ax.set_xticks(range(len(difficulty_order)))
    ax.set_xticklabels(difficulty_order, fontsize=11)
    ax.set_ylabel("Mean Runtime (seconds)", fontsize=11)
    ax.set_title("Runtime vs. Difficulty Tier", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_difficulty_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig4_difficulty_scaling.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: list[dict], configs: list[str]) -> None:
    by_config = group_by(results, "config")
    header = f"{'Config':<30} {'Solved':>6} {'MeanLen':>8} {'MeanNodes':>12} {'MeanTime':>10} {'PeakMB':>8}"
    print("\n" + header)
    print("-" * len(header))
    for cfg in configs:
        rows = by_config.get(cfg, [])
        s_rows = solved_rows(rows)
        if not s_rows:
            print(f"{cfg:<30} {'0/' + str(len(rows)):>6}")
            continue
        ml = statistics.mean([r["solution_length"] for r in s_rows])
        mn = statistics.mean([r["nodes_expanded"] for r in s_rows])
        mt = statistics.mean([r["elapsed_seconds"] for r in s_rows])
        mp = statistics.mean([r["peak_memory_mb"] for r in s_rows])
        solve_str = f"{len(s_rows)}/{len(rows)}"
        print(
            f"{cfg:<30} {solve_str:>6} {ml:>8.1f} {mn:>12,.0f} {mt:>10.3f} {mp:>8.1f}"
        )
