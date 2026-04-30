"""
benchmark.py - Benchmark suite generation and experimental harness.

Generates reproducible solvable 15-puzzle instances, optionally
stratified by Manhattan distance into easy/medium/hard tiers, runs
multiple solvers on each, and serialises results to JSON.
"""

from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import Callable

from puzzle import GOAL_STATE, NEIGHBOURS, is_solvable, manhattan_distance
from solvers import SolverResult, astar, idastar


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def random_walk(seed: int, shuffle_moves: int = 80) -> tuple[int, ...]:
    """Generate a solvable 15-puzzle by performing *shuffle_moves* random
    legal moves from the goal state. Solvability is guaranteed by
    construction; an immediate-reverse check reduces shallow walks.
    """
    rng = random.Random(seed)
    state = list(GOAL_STATE)
    blank = state.index(0)

    prev_blank = -1
    for _ in range(shuffle_moves):
        nbrs = [nb for nb in NEIGHBOURS[blank] if nb != prev_blank]
        if not nbrs:
            nbrs = list(NEIGHBOURS[blank])
        nb = rng.choice(nbrs)
        state[blank], state[nb] = state[nb], state[blank]
        prev_blank, blank = blank, nb
    return tuple(state)


# Backward-compatible alias used in earlier milestones.
generate_instance = random_walk


def classify_difficulty(state: tuple[int, ...]) -> str:
    """Quick difficulty estimate from Manhattan distance (a lower bound
    on the optimal solution length)."""
    md = manhattan_distance(state)
    if md <= 24:
        return "easy"
    elif md <= 36:
        return "medium"
    else:
        return "hard"


def generate_benchmark(
    n: int = 100,
    seed: int = 42,
    save_path: Path | None = None,
    stratified: bool = True,
) -> list[tuple[int, ...]]:
    """Return *n* reproducible solvable instances.

    With *stratified=True* (default), instances are drawn so the suite
    is roughly balanced across easy/medium/hard tiers. The shuffle
    length is varied to reach all three tiers; instances landing in an
    over-quota tier are rejected and re-drawn.
    """
    rng = random.Random(seed)
    instances: list[tuple[int, ...]] = []

    if not stratified:
        attempt = 0
        while len(instances) < n:
            shuffle_len = rng.randint(50, 120)
            inst = random_walk(seed * 1000 + attempt, shuffle_moves=shuffle_len)
            assert is_solvable(inst)
            instances.append(inst)
            attempt += 1
    else:
        target = {
            "easy":   n // 3,
            "medium": n // 3,
            "hard":   n - 2 * (n // 3),
        }
        counts = {"easy": 0, "medium": 0, "hard": 0}
        attempt = 0
        # Map difficulty tiers to shuffle-length distributions that tend
        # to produce them; this is a sampling heuristic, not a guarantee.
        shuffle_for: dict[str, tuple[int, int]] = {
            "easy":   (15, 35),
            "medium": (40, 80),
            "hard":   (90, 200),
        }
        # Round-robin across tiers we still need
        max_attempts = n * 200
        while sum(counts.values()) < n and attempt < max_attempts:
            attempt += 1
            # Pick the tier most under quota
            needed = sorted(
                ((target[t] - counts[t], t) for t in target),
                reverse=True,
            )
            tier = needed[0][1]
            lo, hi = shuffle_for[tier]
            shuffle_len = rng.randint(lo, hi)
            inst = random_walk(seed * 1000 + attempt, shuffle_moves=shuffle_len)
            assert is_solvable(inst)
            actual_tier = classify_difficulty(inst)
            if counts[actual_tier] < target[actual_tier]:
                instances.append(inst)
                counts[actual_tier] += 1

        if len(instances) < n:  # fallback for stragglers
            while len(instances) < n:
                attempt += 1
                shuffle_len = rng.randint(50, 120)
                inst = random_walk(seed * 1000 + attempt, shuffle_moves=shuffle_len)
                assert is_solvable(inst)
                instances.append(inst)

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
        name            : str    - human-readable label
        algorithm       : "astar" | "idastar"
        heuristic       : callable
        heuristic_name  : str
        weight          : float  (for weighted variants; default 1.0)
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
                    end=" ... ",
                    flush=True,
                )

            if algo == "astar":
                res: SolverResult = astar(
                    instance, h,
                    weight=w,
                    heuristic_name=h_name,
                    time_limit=time_limit,
                )
            else:  # idastar (handles both IDA* and Weighted IDA* via weight)
                res = idastar(
                    instance, h,
                    heuristic_name=h_name,
                    weight=w,
                    time_limit=time_limit,
                )

            row = {
                "config": cfg_name,
                "instance_idx": idx,
                "difficulty": classify_difficulty(instance),
                "manhattan": manhattan_distance(instance),
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
                print(
                    f"{status}, {res.nodes_expanded:,} nodes, "
                    f"{res.elapsed_seconds:.2f}s"
                )

            done += 1

    return results


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
