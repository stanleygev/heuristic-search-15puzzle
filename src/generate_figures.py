"""generate_figures.py - run analysis on existing results.json."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis import (
    load_results, print_summary_table, write_summary_csv,
    plot_algorithm_comparison, plot_weighted_tradeoff,
    plot_difficulty_scaling, plot_heuristic_quality, plot_memory_comparison,
)

PROJ_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    results = load_results(PROJ_ROOT / "results" / "results.json")
    figures_dir = PROJ_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg_names = [
        "A*+Manhattan", "A*+LinearConflict",
        "A*+PDB(5-5-5)", "IDA*+PDB(5-5-5)",
        "WeightedA*(w=1.25)", "WeightedA*(w=1.5)", "WeightedA*(w=2.0)",
        "WIDA*(w=1.5)",
    ]
    print_summary_table(results, cfg_names)
    write_summary_csv(results, cfg_names, PROJ_ROOT / "results" / "summary.csv")

    comparison_cfgs = [
        "A*+Manhattan", "A*+LinearConflict",
        "A*+PDB(5-5-5)", "IDA*+PDB(5-5-5)",
    ]
    plot_algorithm_comparison(results, comparison_cfgs, figures_dir)
    plot_difficulty_scaling(results, comparison_cfgs, figures_dir)
    plot_memory_comparison(results, comparison_cfgs, figures_dir)
    plot_weighted_tradeoff(
        results,
        baseline_config="A*+PDB(5-5-5)",
        weighted_configs=[
            ("WeightedA*(w=1.25)", 1.25),
            ("WeightedA*(w=1.5)", 1.50),
            ("WeightedA*(w=2.0)", 2.00),
        ],
        output_dir=figures_dir,
    )
    plot_heuristic_quality(results, "A*+PDB(5-5-5)", figures_dir)
    print("\nFigures written to", figures_dir)


if __name__ == "__main__":
    main()
