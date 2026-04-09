"""
solvers.py - Search algorithm implementations for the 15-puzzle.

Algorithms
----------
* A* with arbitrary heuristic
* IDA* with arbitrary heuristic
* Weighted A* (w * h(n))

Each solver returns a SolverResult dataclass containing the solution
path, number of nodes expanded, peak memory, and wall-clock time.
"""

from __future__ import annotations
import heapq
import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Optional

from puzzle import GOAL_STATE, get_successors, encode_state, manhattan_distance


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Outcome of a single solver run."""
    solved: bool
    solution_length: int                   # number of moves (-1 if not solved)
    nodes_expanded: int
    peak_memory_mb: float
    elapsed_seconds: float
    algorithm: str
    heuristic_name: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

def astar(
    start: tuple[int, ...],
    heuristic: Callable[[tuple[int, ...]], int],
    weight: float = 1.0,
    memory_limit_mb: float = 2048.0,
    heuristic_name: str = "h",
    time_limit: float = 300.0,
) -> SolverResult:
    """A* (or weighted A*) search.

    Parameters
    ----------
    start : initial state (tuple of 16 ints)
    heuristic : admissible heuristic callable
    weight : inflation factor; 1.0 = standard A*
    memory_limit_mb : abort if peak RSS exceeds this
    heuristic_name : label used in SolverResult
    time_limit : wall-clock cutoff in seconds
    """
    algorithm = "A*" if weight == 1.0 else f"Weighted-A*(w={weight})"

    tracemalloc.start()
    t0 = time.perf_counter()

    h0 = heuristic(start)
    # heap entry: (f, g, encoded_state, state)
    open_heap: list[tuple[float, int, int, tuple[int, ...]]] = [
        (weight * h0, 0, encode_state(start), start)
    ]
    g_map: dict[tuple[int, ...], int] = {start: 0}
    nodes_expanded = 0

    while open_heap:
        # Memory / time guard
        if nodes_expanded % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            _, peak_kb = tracemalloc.get_traced_memory()
            peak_mb = peak_kb / 1024
            if peak_mb > memory_limit_mb or elapsed > time_limit:
                tracemalloc.stop()
                return SolverResult(
                    solved=False,
                    solution_length=-1,
                    nodes_expanded=nodes_expanded,
                    peak_memory_mb=peak_mb,
                    elapsed_seconds=elapsed,
                    algorithm=algorithm,
                    heuristic_name=heuristic_name,
                    weight=weight,
                )

        f, g, _, state = heapq.heappop(open_heap)

        if g > g_map.get(state, math.inf):
            continue  # stale entry

        if state == GOAL_STATE:
            _, peak_kb = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return SolverResult(
                solved=True,
                solution_length=g,
                nodes_expanded=nodes_expanded,
                peak_memory_mb=peak_kb / 1024,
                elapsed_seconds=time.perf_counter() - t0,
                algorithm=algorithm,
                heuristic_name=heuristic_name,
                weight=weight,
            )

        nodes_expanded += 1
        for successor, cost in get_successors(state):
            new_g = g + cost
            if new_g < g_map.get(successor, math.inf):
                g_map[successor] = new_g
                h = heuristic(successor)
                f_new = new_g + weight * h
                heapq.heappush(open_heap, (f_new, new_g, encode_state(successor), successor))

    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return SolverResult(
        solved=False,
        solution_length=-1,
        nodes_expanded=nodes_expanded,
        peak_memory_mb=peak_kb / 1024,
        elapsed_seconds=time.perf_counter() - t0,
        algorithm=algorithm,
        heuristic_name=heuristic_name,
        weight=weight,
    )


# ---------------------------------------------------------------------------
# IDA* search
# ---------------------------------------------------------------------------

def idastar(
    start: tuple[int, ...],
    heuristic: Callable[[tuple[int, ...]], int],
    heuristic_name: str = "h",
    time_limit: float = 300.0,
) -> SolverResult:
    """IDA* (iterative-deepening A*) search with linear memory usage."""

    tracemalloc.start()
    t0 = time.perf_counter()
    nodes_expanded = 0
    found = False
    solution_length = -1

    # Internal DFS that returns (found, min_exceeded_threshold)
    def _dfs(
        state: tuple[int, ...],
        g: int,
        threshold: int,
        prev_state: Optional[tuple[int, ...]]
    ) -> tuple[bool, int]:
        nonlocal nodes_expanded, found, t0

        h = heuristic(state)
        f = g + h
        if f > threshold:
            return False, f
        if state == GOAL_STATE:
            return True, g

        nodes_expanded += 1
        minimum = math.inf
        for successor, cost in get_successors(state):
            if successor == prev_state:
                continue  # no-reversal pruning
            if time.perf_counter() - t0 > time_limit:
                return False, -1  # signal timeout via negative value
            solved, val = _dfs(successor, g + cost, threshold, state)
            if solved:
                return True, val
            if val < 0:  # timeout propagation
                return False, -1
            if val < minimum:
                minimum = val
        return False, minimum

    threshold = heuristic(start)
    while True:
        if time.perf_counter() - t0 > time_limit:
            break
        solved, result = _dfs(start, 0, threshold, None)
        if solved:
            found = True
            solution_length = result
            break
        if result < 0 or result == math.inf:
            break  # timeout or no solution
        threshold = result

    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return SolverResult(
        solved=found,
        solution_length=solution_length,
        nodes_expanded=nodes_expanded,
        peak_memory_mb=peak_kb / 1024,
        elapsed_seconds=time.perf_counter() - t0,
        algorithm="IDA*",
        heuristic_name=heuristic_name,
        weight=1.0,
    )
