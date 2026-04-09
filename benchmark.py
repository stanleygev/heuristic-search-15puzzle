"""
benchmark.py - Benchmark suite generation and experimental harness.

Generates 100 reproducible solvable 15-puzzle instances stratified by
difficulty, runs multiple solvers, collects metrics, and serialises
results to JSON for downstream analysis.
"""

from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import Callable

from puzzle import GOAL_STATE, is_solvable, manhattan_distance, linear_conflict
from solvers import SolverResult, astar, idastar


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_instance(seed: int, shuffle_moves: int = 80) -> tuple[int, ...]:
    """Generate a solvable 15-puzzle instance by making *shuffle_moves*
    random legal moves from the goal state.  Using moves (rather than
    random permutation + solvability check) guarantees solvability.
    """
    rng = random.Random(seed)
    state = list(GOAL_STATE)
    blank = state.index(0)
    _NEIGHBOURS: list[list[int]] = []
    for i in range(16):
        r, c = i // 4, i % 4
        nb = []
        if r > 0: nb.append(i - 4)
        if r < 3: nb.append(i + 4)
        if c > 0: nb.append(i - 1)
        if c < 3: nb.append(i + 1)
        _NEIGHBOURS.append(nb)

    prev_blank = -1
    for _ in range(shuffle_moves):
        nbrs = [nb for nb in _NEIGHBOURS[blank] if nb != prev_blank]
        if not nbrs:
            nbrs = _NEIGHBOURS[blank]
        nb = rng.choice(nbrs)
        state[blank], state[nb] = state[nb], state[blank]
        prev_blank, blank = blank, nb
    return tuple(state)


def classify_difficulty(state: tuple[int, ...]) -> str:
    """Quick difficulty estimate from Manhattan distance."""
    md = manhattan_distance(state)
    if md <= 30:
        return "easy"
    elif md <= 45:
        return "medium"
    else:
        return "hard"


def generate_benchmark(
    n: int = 100,
    seed: int = 42,
    save_path: Path | None = None,
) -> list[tuple[int, ...]]:
    """Return *n* reproducible solvable instances.

    Instances are generated with increasing shuffle lengths so that the
    benchmark covers a range of difficulties.
    """
    rng = random.Random(seed)
    instances: list[tuple[int, ...]] = []
    attempt = 0
    while len(instances) < n:
        shuffle_len = rng.randint(50, 120)
        inst = generate_instance(seed * 1000 + attempt, shuffle_moves=shuffle_len)
        assert is_solvable(inst), "Generated unsolvable instance!"
        instances.append(inst)
        attempt += 1

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump([list(s) for s in instances], f, indent=2)

    return instances


def load_benchmark(path: Path) -> list[tuple[int, ...]]:
    with open(path) as f:
        return [tuple(s) for s in json.load(f)]


# ---------------------------------------------------------------------------
# Experimental runner
# ---------------------------------------------------------------------------

def run_experiment(
    instances: list[tuple[int, ...]],
    solver_configs: list[dict],
    time_limit: float = 60.0,
    verbose: bool = True,
) -> list[dict]:
    """Run each solver config on each instance and collect metrics.

    *solver_configs* is a list of dicts with keys:
        name       : str - human-readable label
        algorithm  : "astar" | "idastar"
        heuristic  : callable
        heuristic_name : str
        weight     : float (for weighted A*)
    """
    results = []
    total = len(instances) * len(solver_configs)
    done = 0

    for cfg in solver_configs:
        algo = cfg["algorithm"]
        h = cfg["heuristic"]
        h_name = cfg.get("heuristic_name", "h")
        w = cfg.get("weight", 1.0)
        cfg_name = cfg["name"]

        for idx, instance in enumerate(instances):
            if verbose:
                diff = classify_difficulty(instance)
                print(
                    f"  [{done+1}/{total}] {cfg_name} | instance {idx+1} ({diff})",
                    end=" … ",
                    flush=True,
                )
            t0 = time.perf_counter()
            if algo == "astar":
                res: SolverResult = astar(
                    instance, h,
                    weight=w,
                    heuristic_name=h_name,
                    time_limit=time_limit,
                )
            else:
                res = idastar(
                    instance, h,
                    heuristic_name=h_name,
                    time_limit=time_limit,
                )

            row = {
                "config": cfg_name,
                "instance_idx": idx,
                "difficulty": classify_difficulty(instance),
                "solved": res.solved,
                "solution_length": res.solution_length,
                "nodes_expanded": res.nodes_expanded,
                "peak_memory_mb": round(res.peak_memory_mb, 3),
                "elapsed_seconds": round(res.elapsed_seconds, 4),
                "algorithm": res.algorithm,
                "heuristic": h_name,
                "weight": w,
            }
            results.append(row)

            if verbose:
                status = f"len={res.solution_length}" if res.solved else "TIMEOUT"
                print(f"{status}, {res.nodes_expanded:,} nodes, {res.elapsed_seconds:.2f}s")

            done += 1

    return results


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
