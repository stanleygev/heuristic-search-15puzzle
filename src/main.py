"""
main.py - Main entry point for the 15-puzzle solver experiments.

Modes
-----
test       : run baseline test cases on hand-picked instances (fast)
quick      : build / load PDB and benchmark a small subset of solvers
benchmark  : full benchmark with all solvers and figure generation
stats      : load existing results.json and run paired Wilcoxon analysis

Examples
--------
    python main.py --mode test
    python main.py --mode quick --n 10
    python main.py --mode benchmark --n 60 --partition 5-5-5 --time-limit 60
    python main.py --mode stats   --results results/results.json
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure src/ is on path when running from anywhere
sys.path.insert(0, str(Path(__file__).parent))

from puzzle import (
    GOAL_STATE, manhattan_distance, linear_conflict, pretty_print, is_solvable,
)
from solvers import astar, idastar
from benchmark import (
    generate_benchmark, load_benchmark, run_experiment, save_results,
)
from analysis import (
    load_results, print_summary_table, write_summary_csv,
    plot_algorithm_comparison, plot_weighted_tradeoff,
    plot_difficulty_scaling, plot_heuristic_quality, plot_memory_comparison,
)


# ---------------------------------------------------------------------------
# Test cases (all solvable; previous milestone had an unsolvable example)
# ---------------------------------------------------------------------------

TEST_CASES = [
    ("Goal state (trivial)",
     (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)),
    ("One move from goal",
     (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15)),
    ("Two moves from goal",
     (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15)),
    ("5-step random walk (easy)",
     (1, 2, 4, 0, 5, 6, 3, 7, 9, 10, 11, 8, 13, 14, 15, 12)),
    ("10-step random walk",
     (1, 2, 7, 3, 5, 0, 4, 8, 9, 6, 10, 11, 13, 14, 15, 12)),
    ("20-step random walk (medium)",
     (1, 10, 0, 2, 5, 11, 6, 3, 9, 8, 15, 4, 13, 14, 12, 7)),
]


def run_test_cases(use_lc: bool = True) -> bool:
    print("=" * 60)
    print("TEST CASES - A* with Manhattan Distance")
    print("=" * 60)
    all_passed = True
    for desc, state in TEST_CASES:
        assert is_solvable(state), f"Test '{desc}' is unsolvable!"
        print(f"\n{desc}")
        print(pretty_print(state))
        result = astar(state, manhattan_distance,
                       heuristic_name="Manhattan", time_limit=30.0)
        status = "[PASS]" if result.solved else "[FAIL]"
        print(f"  {status} Length={result.solution_length} | "
              f"Nodes={result.nodes_expanded:,} | Time={result.elapsed_seconds:.4f}s")
        if not result.solved:
            all_passed = False

    if use_lc:
        print("\n" + "=" * 60)
        print("TEST CASES - A* with Linear Conflict")
        print("=" * 60)
        for desc, state in TEST_CASES:
            print(f"\n{desc}")
            result = astar(state, linear_conflict,
                           heuristic_name="LinearConflict", time_limit=30.0)
            status = "[PASS]" if result.solved else "[FAIL]"
            print(f"  {status} Length={result.solution_length} | "
                  f"Nodes={result.nodes_expanded:,} | Time={result.elapsed_seconds:.4f}s")
            if not result.solved:
                all_passed = False

    print("\n" + ("All test cases passed." if all_passed else "Some tests FAILED."))
    return all_passed


# ---------------------------------------------------------------------------
# Quick mode
# ---------------------------------------------------------------------------

def run_quick(n: int, seed: int, time_limit: float, cache_dir: Path) -> None:
    from pdb import DisjointPDB, PARTITION_5_5_5

    print(f"\nGenerating {n} stratified benchmark instances (seed={seed}) ...")
    instances = generate_benchmark(n=n, seed=seed)

    print("Building / loading PDB (5-5-5) ...")
    pdb = DisjointPDB(PARTITION_5_5_5, cache_dir=cache_dir, verbose=True)

    configs = [
        {"name": "A*+Manhattan",      "algorithm": "astar",   "heuristic": manhattan_distance, "heuristic_name": "Manhattan",     "weight": 1.0},
        {"name": "A*+LinearConflict", "algorithm": "astar",   "heuristic": linear_conflict,    "heuristic_name": "LinearConflict","weight": 1.0},
        {"name": "A*+PDB(5-5-5)",     "algorithm": "astar",   "heuristic": pdb,                "heuristic_name": "PDB(5-5-5)",    "weight": 1.0},
        {"name": "IDA*+PDB(5-5-5)",   "algorithm": "idastar", "heuristic": pdb,                "heuristic_name": "PDB(5-5-5)",    "weight": 1.0},
    ]

    print(f"\nRunning {len(configs)} solvers on {n} instances ...\n")
    results = run_experiment(instances, configs, time_limit=time_limit, verbose=True)
    print_summary_table(results, [c["name"] for c in configs])


# ---------------------------------------------------------------------------
# Full benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(
    n: int,
    seed: int,
    time_limit: float,
    cache_dir: Path,
    results_path: Path,
    figures_dir: Path,
    partition_name: str,
) -> None:
    from pdb import DisjointPDB, PARTITION_5_5_5, PARTITION_6_5_4, PARTITION_7_8

    partition_map = {
        "5-5-5": PARTITION_5_5_5,
        "6-5-4": PARTITION_6_5_4,
        "7-8":   PARTITION_7_8,
    }
    partition = partition_map[partition_name]

    print(f"\nGenerating {n} stratified benchmark instances (seed={seed}) ...")
    bench_path = Path(__file__).resolve().parent.parent / "data" / "benchmark.json"
    instances = generate_benchmark(n=n, seed=seed, save_path=bench_path)

    print(f"\nBuilding / loading Disjoint PDB ({partition_name}) ...")
    pdb_main = DisjointPDB(partition, cache_dir=cache_dir, verbose=True)

    # Always include 5-5-5 for partition comparison even if user picked a
    # different primary partition.
    if partition_name != "5-5-5":
        print("\nBuilding / loading secondary PDB (5-5-5) for comparison ...")
        pdb_555 = DisjointPDB(PARTITION_5_5_5, cache_dir=cache_dir, verbose=True)
    else:
        pdb_555 = pdb_main

    configs = [
        # Heuristic comparison
        {"name": "A*+Manhattan",       "algorithm": "astar",   "heuristic": manhattan_distance, "heuristic_name": "Manhattan",        "weight": 1.0},
        {"name": "A*+LinearConflict",  "algorithm": "astar",   "heuristic": linear_conflict,    "heuristic_name": "LinearConflict",   "weight": 1.0},
        {"name": "A*+PDB(5-5-5)",      "algorithm": "astar",   "heuristic": pdb_555,            "heuristic_name": "PDB(5-5-5)",       "weight": 1.0},
        # IDA*
        {"name": f"IDA*+PDB({partition_name})", "algorithm": "idastar", "heuristic": pdb_main,   "heuristic_name": f"PDB({partition_name})", "weight": 1.0},
        # Weighted A*
        {"name": "WeightedA*(w=1.25)", "algorithm": "astar",   "heuristic": pdb_main,           "heuristic_name": f"PDB({partition_name})", "weight": 1.25},
        {"name": "WeightedA*(w=1.5)",  "algorithm": "astar",   "heuristic": pdb_main,           "heuristic_name": f"PDB({partition_name})", "weight": 1.50},
        {"name": "WeightedA*(w=2.0)",  "algorithm": "astar",   "heuristic": pdb_main,           "heuristic_name": f"PDB({partition_name})", "weight": 2.00},
        # Weighted IDA*  (5th enhancement)
        {"name": "WIDA*(w=1.5)",       "algorithm": "idastar", "heuristic": pdb_main,           "heuristic_name": f"PDB({partition_name})", "weight": 1.50},
    ]

    if partition_name != "5-5-5":
        # Second partition for partition-strategy comparison
        configs.insert(
            3,
            {"name": f"A*+PDB({partition_name})", "algorithm": "astar",
             "heuristic": pdb_main, "heuristic_name": f"PDB({partition_name})", "weight": 1.0},
        )

    print(f"\nRunning {len(configs)} solvers on {n} instances ...\n")
    results = run_experiment(instances, configs, time_limit=time_limit, verbose=True)

    save_results(results, results_path)
    print(f"\nResults saved to {results_path}")

    cfg_names = [c["name"] for c in configs]
    print_summary_table(results, cfg_names)
    write_summary_csv(results, cfg_names, results_path.parent / "summary.csv")

    # Figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison_cfgs = [
        c for c in [
            "A*+Manhattan", "A*+LinearConflict", "A*+PDB(5-5-5)",
            f"A*+PDB({partition_name})" if partition_name != "5-5-5" else None,
            f"IDA*+PDB({partition_name})",
        ] if c
    ]
    plot_algorithm_comparison(results, comparison_cfgs, figures_dir)
    plot_weighted_tradeoff(
        results,
        baseline_config="A*+PDB(5-5-5)",
        weighted_configs=[
            ("WeightedA*(w=1.25)", 1.25),
            ("WeightedA*(w=1.5)",  1.50),
            ("WeightedA*(w=2.0)",  2.00),
            ("WIDA*(w=1.5)",       1.50),
        ],
        output_dir=figures_dir,
    )
    plot_difficulty_scaling(results, comparison_cfgs, figures_dir)
    plot_heuristic_quality(results, "A*+PDB(5-5-5)", figures_dir)
    plot_memory_comparison(results, comparison_cfgs, figures_dir)
    print(f"\nFigures saved to {figures_dir}/")


# ---------------------------------------------------------------------------
# Stats mode
# ---------------------------------------------------------------------------

def run_stats(results_path: Path, figures_dir: Path) -> None:
    from stats import paired_wilcoxon_table, format_comparison_table
    results = load_results(results_path)
    configs = sorted({r["config"] for r in results})
    print(f"Loaded {len(results):,} rows across {len(configs)} configs.\n")

    print("=" * 78)
    print("Paired Wilcoxon signed-rank test, Bonferroni-corrected (alpha = 0.05)")
    print("Metric: elapsed_seconds (only instances solved by both solvers)")
    print("=" * 78)
    comps_time = paired_wilcoxon_table(results, configs, metric="elapsed_seconds")
    print(format_comparison_table(comps_time))

    print()
    print("=" * 78)
    print("Same test, metric = nodes_expanded")
    print("=" * 78)
    comps_nodes = paired_wilcoxon_table(results, configs, metric="nodes_expanded")
    print(format_comparison_table(comps_nodes))

    # Save tables to disk
    figures_dir.mkdir(parents=True, exist_ok=True)
    with open(figures_dir.parent / "wilcoxon_time.txt", "w") as f:
        f.write(format_comparison_table(comps_time))
    with open(figures_dir.parent / "wilcoxon_nodes.txt", "w") as f:
        f.write(format_comparison_table(comps_nodes))
    print(f"\nWilcoxon tables saved to {figures_dir.parent}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="15-Puzzle Solver Experiments")
    parser.add_argument("--mode",
                        choices=["test", "quick", "benchmark", "stats"],
                        default="test")
    parser.add_argument("--n",          type=int,   default=100)
    parser.add_argument("--cache",      type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data" / "pdb_cache"))
    parser.add_argument("--results",    type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results" / "results.json"))
    parser.add_argument("--figures",    type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results" / "figures"))
    parser.add_argument("--partition",  choices=["7-8", "5-5-5", "6-5-4"],
                        default="5-5-5")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        dest="time_limit")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    if args.mode == "test":
        run_test_cases()
    elif args.mode == "quick":
        run_quick(
            n=args.n,
            seed=args.seed,
            time_limit=args.time_limit,
            cache_dir=Path(args.cache),
        )
    elif args.mode == "benchmark":
        run_benchmark(
            n=args.n,
            seed=args.seed,
            time_limit=args.time_limit,
            cache_dir=Path(args.cache),
            results_path=Path(args.results),
            figures_dir=Path(args.figures),
            partition_name=args.partition,
        )
    elif args.mode == "stats":
        run_stats(
            results_path=Path(args.results),
            figures_dir=Path(args.figures),
        )


if __name__ == "__main__":
    main()
