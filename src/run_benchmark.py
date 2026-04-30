"""
run_benchmark.py - Incremental benchmark runner.

Runs ONE solver config across all instances, saving results incrementally.
Idempotent: re-running with the same config name skips already-completed
work. This lets the long benchmark be split across multiple bash sessions.

Usage:
    python run_benchmark.py <config_name> [n] [time_limit]

Configs:
    manhattan, linearconflict, astar_pdb, idastar_pdb,
    weighted_125, weighted_15, weighted_20, wida_15
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from puzzle import manhattan_distance, linear_conflict
from benchmark import generate_benchmark, classify_difficulty
from solvers import astar, idastar
from pdb import DisjointPDB, PARTITION_5_5_5

PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJ_ROOT / "results" / "results.json"
BENCH_PATH = PROJ_ROOT / "data" / "benchmark.json"
PDB_CACHE = PROJ_ROOT / "data" / "pdb_cache"


CONFIG_TABLE = {
    # name              -> (config_label, algorithm, needs_pdb, weight)
    "manhattan":        ("A*+Manhattan",      "astar",   False, 1.0),
    "linearconflict":   ("A*+LinearConflict", "astar",   False, 1.0),
    "astar_pdb":        ("A*+PDB(5-5-5)",     "astar",   True,  1.0),
    "idastar_pdb":      ("IDA*+PDB(5-5-5)",   "idastar", True,  1.0),
    "weighted_125":     ("WeightedA*(w=1.25)","astar",   True,  1.25),
    "weighted_15":      ("WeightedA*(w=1.5)", "astar",   True,  1.50),
    "weighted_20":      ("WeightedA*(w=2.0)", "astar",   True,  2.00),
    "wida_15":          ("WIDA*(w=1.5)",      "idastar", True,  1.50),
}


def load_existing() -> list[dict]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_atomic(results: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(RESULTS_PATH)


def main(config_key: str, n: int, time_limit: float) -> None:
    if config_key not in CONFIG_TABLE:
        raise SystemExit(f"Unknown config: {config_key}. "
                         f"Choose from: {list(CONFIG_TABLE)}")
    cfg_label, algo, needs_pdb, weight = CONFIG_TABLE[config_key]

    existing = load_existing()
    done_for_cfg = {r["instance_idx"] for r in existing if r["config"] == cfg_label}
    print(f"Existing results: {len(existing)} rows total, "
          f"{len(done_for_cfg)} already done for {cfg_label}")

    instances = generate_benchmark(n=n, seed=42, save_path=BENCH_PATH)
    print(f"Loaded {n} instances")

    # Heuristic
    if needs_pdb:
        print("Loading PDB(5-5-5) ...")
        h = DisjointPDB(PARTITION_5_5_5,
                        cache_dir=PDB_CACHE,
                        verbose=False)
        h_name = "PDB(5-5-5)"
    elif config_key == "manhattan":
        h = manhattan_distance
        h_name = "Manhattan"
    elif config_key == "linearconflict":
        h = linear_conflict
        h_name = "LinearConflict"

    results = list(existing)
    save_every = 3  # checkpoint after every 3 instances
    pending = 0

    for idx, state in enumerate(instances):
        if idx in done_for_cfg:
            continue

        diff = classify_difficulty(state)
        print(f"  [{idx+1}/{n}] {cfg_label} | inst {idx+1} ({diff})", end=" ... ", flush=True)

        if algo == "astar":
            res = astar(state, h, weight=weight,
                        heuristic_name=h_name, time_limit=time_limit)
        else:
            res = idastar(state, h, weight=weight,
                          heuristic_name=h_name, time_limit=time_limit)

        from puzzle import manhattan_distance as md_func
        row = {
            "config": cfg_label,
            "instance_idx": idx,
            "difficulty": diff,
            "manhattan": md_func(state),
            "solved": res.solved,
            "solution_length": res.solution_length,
            "nodes_expanded": res.nodes_expanded,
            "peak_memory_mb": round(res.peak_memory_mb, 3),
            "elapsed_seconds": round(res.elapsed_seconds, 4),
            "algorithm": res.algorithm,
            "heuristic": h_name,
            "weight": weight,
        }
        results.append(row)
        pending += 1

        status = f"len={res.solution_length}" if res.solved else "TIMEOUT"
        print(f"{status}, {res.nodes_expanded:,} nodes, {res.elapsed_seconds:.2f}s")

        if pending >= save_every:
            save_atomic(results)
            pending = 0

    if pending:
        save_atomic(results)

    print(f"\nDone with {cfg_label}.  Results: {len(results)} total rows.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cfg = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    main(cfg, n, tl)
