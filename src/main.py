"""
main.py - Main entry point for the 15-puzzle solver experiments.

Usage
-----
Run baseline test cases only (fast, no PDB needed):
    python main.py --mode test

Run full benchmark experiments (PDB build + 100 instances):
    python main.py --mode benchmark

Run only heuristic comparison on a small subset:
    python main.py --mode quick --n 10

Options
-------
--mode    : test | quick | benchmark  (default: test)
--n       : number of instances for quick/benchmark  (default: 100)
--cache   : directory for cached PDB files  (default: data/pdb_cache)
--results : output path for JSON results  (default: results/results.json)
--figures : output directory for figures  (default: results/figures)
--partition : 7-8 | 5-5-5 | 6-5-4  (PDB partition to use; default: 5-5-5)
--time-limit : per-instance time limit in seconds  (default: 60)
--seed    : random seed for benchmark generation  (default: 42)
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from puzzle import GOAL_STATE, manhattan_distance, linear_conflict, pretty_print
from solvers import astar, idastar
from benchmark import generate_benchmark, load_benchmark, run_experiment, save_results
from analysis import (
    load_results, print_summary_table, plot_algorithm_comparison,
    plot_weighted_tradeoff, plot_difficulty_scaling,
)


# ---------------------------------------------------------------------------
# Test-case verification
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (description, state)
    (
        "Goal state (trivial)",
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0),
    ),
    (
        "One move from goal",
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15),
    ),
    (
        "Easy instance (~8 moves)",
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15),
    ),
    (
        "Medium instance",
        (5, 1, 3, 4, 9, 2, 7, 8, 13, 6, 11, 12, 0, 10, 14, 15),
    ),
    (
        "Harder instance",
        (2, 8, 3, 4, 1, 6, 7, 12, 5, 9, 10, 11, 13, 14, 15, 0),
    ),
]


def run_test_cases(use_lc: bool = True) -> None:
    print("=" * 60)
    print("TEST CASES - A* with Manhattan Distance")
    print("=" * 60)
    all_passed = True
    for desc, state in TEST_CASES:
        print(f"\n{desc}")
        print(pretty_print(state))
        result = astar(state, manhattan_distance, heuristic_name="Manhattan", time_limit=30.0)
        status = "✓ SOLVED" if result.solved else "✗ FAILED"
        print(
            f"  {status} | Length={result.solution_length} | "
            f"Nodes={result.nodes_expanded:,} | Time={result.elapsed_seconds:.4f}s"
        )
        if not result.solved:
            all_passed = False

    if use_lc:
        print("\n" + "=" * 60)
        print("TEST CASES - A* with Linear Conflict")
        print("=" * 60)
        for desc, state in TEST_CASES:
            print(f"\n{desc}")
            result = astar(state, linear_conflict, heuristic_name="LinearConflict", time_limit=30.0)
            status = "✓ SOLVED" if result.solved else "✗ FAILED"
            print(
                f"  {status} | Length={result.solution_length} | "
                f"Nodes={result.nodes_expanded:,} | Time={result.elapsed_seconds:.4f}s"
            )
            if not result.solved:
                all_passed = False

    print("\n" + ("All test cases passed ✓" if all_passed else "Some test cases FAILED ✗"))
    return all_passed


# ---------------------------------------------------------------------------
# Quick mode (subset benchmark)
# ---------------------------------------------------------------------------

def run_quick(n: int, seed: int, time_limit: float, cache_dir: Path) -> None:
    from pdb import DisjointPDB, PARTITION_5_5_5

    print(f"\nGenerating {n} benchmark instances (seed={seed}) …")
    instances = generate_benchmark(n=n, seed=seed)

    print("Building PDB (5-5-5) …")
    pdb = DisjointPDB(PARTITION_5_5_5, cache_dir=cache_dir, verbose=True)

    configs = [
        {"name": "A*+Manhattan",      "algorithm": "astar",  "heuristic": manhattan_distance, "heuristic_name": "Manhattan",     "weight": 1.0},
        {"name": "A*+LinearConflict", "algorithm": "astar",  "heuristic": linear_conflict,    "heuristic_name": "LinearConflict","weight": 1.0},
        {"name": "A*+PDB(5-5-5)",     "algorithm": "astar",  "heuristic": pdb,                "heuristic_name": "PDB(5-5-5)",    "weight": 1.0},
    ]

    print(f"\nRunning {len(configs)} solvers on {n} instances …\n")
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

    print(f"\nGenerating {n} benchmark instances (seed={seed}) …")
    bench_path = Path("data/benchmark.json")
    instances = generate_benchmark(n=n, seed=seed, save_path=bench_path)

    print(f"\nBuilding Disjoint PDB ({partition_name}) …")
    pdb = DisjointPDB(partition, cache_dir=cache_dir, verbose=True)

    print(f"\nBuilding secondary PDB (5-5-5) for partition comparison …")
    pdb_555 = DisjointPDB(PARTITION_5_5_5, cache_dir=cache_dir, verbose=True)

    configs = [
        # Heuristic comparison
        {"name": "A*+Manhattan",       "algorithm": "astar",   "heuristic": manhattan_distance, "heuristic_name": "Manhattan",        "weight": 1.0},
        {"name": "A*+LinearConflict",  "algorithm": "astar",   "heuristic": linear_conflict,    "heuristic_name": "LinearConflict",   "weight": 1.0},
        {"name": "A*+PDB(5-5-5)",      "algorithm": "astar",   "heuristic": pdb_555,            "heuristic_name": "PDB(5-5-5)",       "weight": 1.0},
        {"name": f"A*+PDB({partition_name})", "algorithm": "astar", "heuristic": pdb,           "heuristic_name": f"PDB({partition_name})", "weight": 1.0},
        # IDA*
        {"name": f"IDA*+PDB({partition_name})", "algorithm": "idastar", "heuristic": pdb,       "heuristic_name": f"PDB({partition_name})", "weight": 1.0},
        # Weighted A*
        {"name": "WeightedA*(w=1.25)", "algorithm": "astar",   "heuristic": pdb,                "heuristic_name": f"PDB({partition_name})", "weight": 1.25},
        {"name": "WeightedA*(w=1.5)",  "algorithm": "astar",   "heuristic": pdb,                "heuristic_name": f"PDB({partition_name})", "weight": 1.50},
        {"name": "WeightedA*(w=2.0)",  "algorithm": "astar",   "heuristic": pdb,                "heuristic_name": f"PDB({partition_name})", "weight": 2.00},
    ]

    print(f"\nRunning {len(configs)} solvers on {n} instances …\n")
    results = run_experiment(instances, configs, time_limit=time_limit, verbose=True)

    save_results(results, results_path)
    print(f"\nResults saved to {results_path}")

    print_summary_table(results, [c["name"] for c in configs])

    # Figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison_cfgs = [
        "A*+Manhattan", "A*+LinearConflict", "A*+PDB(5-5-5)",
        f"A*+PDB({partition_name})", f"IDA*+PDB({partition_name})",
    ]
    plot_algorithm_comparison(results, comparison_cfgs, figures_dir)
    plot_weighted_tradeoff(
        results,
        baseline_config=f"A*+PDB({partition_name})",
        weighted_configs=[
            ("WeightedA*(w=1.25)", 1.25),
            ("WeightedA*(w=1.5)",  1.50),
            ("WeightedA*(w=2.0)",  2.00),
        ],
        output_dir=figures_dir,
    )
    plot_difficulty_scaling(results, comparison_cfgs, figures_dir)
    print(f"\nFigures saved to {figures_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="15-Puzzle Solver Experiments")
    parser.add_argument("--mode",       choices=["test", "quick", "benchmark"], default="test")
    parser.add_argument("--n",          type=int,   default=100)
    parser.add_argument("--cache",      type=str,   default="data/pdb_cache")
    parser.add_argument("--results",    type=str,   default="results/results.json")
    parser.add_argument("--figures",    type=str,   default="results/figures")
    parser.add_argument("--partition",  choices=["7-8", "5-5-5", "6-5-4"], default="5-5-5")
    parser.add_argument("--time-limit", type=float, default=60.0, dest="time_limit")
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


if __name__ == "__main__":
    main()
