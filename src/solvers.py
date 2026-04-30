"""
solvers.py - Search algorithms for the 15-puzzle.

Algorithms
----------
* astar()   - A* with arbitrary heuristic; weight=1.0 is plain A*,
              weight>1.0 gives Weighted A* (bounded suboptimal).
* idastar() - IDA* with arbitrary heuristic and no-reversal pruning.
* widastar()- Weighted IDA* (bounded suboptimal); a 5th enhancement
              that combines linear-memory IDA* with an inflated heuristic.

All solvers return a SolverResult dataclass with solution length, nodes
expanded, peak memory, wall-clock elapsed time, and configuration labels.
"""

from __future__ import annotations
import heapq
import math
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Optional

from puzzle import GOAL_STATE, get_successors, encode_state


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Outcome of a single solver run."""
    solved: bool
    solution_length: int                   # number of moves; -1 if not solved
    nodes_expanded: int                    # number of states whose successors were generated
    peak_memory_mb: float                  # peak Python heap allocation observed
    elapsed_seconds: float                 # wall-clock time
    algorithm: str                         # "A*", "IDA*", "Weighted-A*(w=1.5)", ...
    heuristic_name: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# A* / Weighted A*
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
    start          : initial state (16-tuple)
    heuristic      : admissible heuristic callable; weighted variant
                     inflates this by *weight*.
    weight         : >= 1.0; 1.0 yields optimal A*; w > 1 is bounded
                     suboptimal at most w times optimal.
    memory_limit_mb: abort if traced peak memory exceeds this bound
    heuristic_name : label stored in SolverResult
    time_limit     : wall-clock cutoff in seconds
    """
    algorithm = "A*" if weight == 1.0 else f"Weighted-A*(w={weight})"

    tracemalloc.start()
    t0 = time.perf_counter()

    h0 = heuristic(start)
    # Heap entry: (f, g, encoded_state, state). encoded_state is the
    # tie-breaker so tuples don't get compared (which would be slow).
    open_heap: list[tuple[float, int, int, tuple[int, ...]]] = [
        (weight * h0, 0, encode_state(start), start)
    ]
    g_map: dict[tuple[int, ...], int] = {start: 0}
    nodes_expanded = 0

    while open_heap:
        # Memory / time guard, sampled every 10k expansions to amortize cost.
        if nodes_expanded % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_mb = peak_bytes / (1024 * 1024)
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
            continue  # stale entry — better g already recorded

        if state == GOAL_STATE:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return SolverResult(
                solved=True,
                solution_length=g,
                nodes_expanded=nodes_expanded,
                peak_memory_mb=peak_bytes / (1024 * 1024),
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
                heapq.heappush(
                    open_heap,
                    (f_new, new_g, encode_state(successor), successor),
                )

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return SolverResult(
        solved=False,
        solution_length=-1,
        nodes_expanded=nodes_expanded,
        peak_memory_mb=peak_bytes / (1024 * 1024),
        elapsed_seconds=time.perf_counter() - t0,
        algorithm=algorithm,
        heuristic_name=heuristic_name,
        weight=weight,
    )


# ---------------------------------------------------------------------------
# IDA* / Weighted IDA*
# ---------------------------------------------------------------------------

def idastar(
    start: tuple[int, ...],
    heuristic: Callable[[tuple[int, ...]], int],
    heuristic_name: str = "h",
    weight: float = 1.0,
    time_limit: float = 300.0,
) -> SolverResult:
    """IDA* search with no-reversal pruning. Uses O(d) memory.

    Setting weight > 1.0 yields Weighted IDA* (WIDA*), a bounded-
    suboptimal linear-memory variant. The first solution found has cost
    at most *weight* times optimal.
    """
    algorithm = "IDA*" if weight == 1.0 else f"Weighted-IDA*(w={weight})"

    tracemalloc.start()
    t0 = time.perf_counter()
    nodes_expanded = 0
    found = False
    solution_length = -1

    # Track the deepest reachable g found so far for the timeout case
    timeout_flag = [False]

    def _dfs(
        state: tuple[int, ...],
        g: int,
        threshold: float,
        prev_state: Optional[tuple[int, ...]],
    ) -> tuple[bool, float]:
        """Depth-first search bounded by f(n) <= threshold.

        Returns (found, next_threshold). When not found, next_threshold
        is the minimum f-value that exceeded the current threshold.
        """
        nonlocal nodes_expanded

        h = heuristic(state)
        f = g + weight * h
        if f > threshold:
            return False, f
        if state == GOAL_STATE:
            return True, g

        nodes_expanded += 1
        # Sampled timeout check
        if (nodes_expanded & 0xFFFF) == 0:
            if time.perf_counter() - t0 > time_limit:
                timeout_flag[0] = True
                return False, math.inf

        minimum = math.inf
        for successor, _ in get_successors(state):
            if successor == prev_state:
                continue  # no-reversal pruning
            solved, val = _dfs(successor, g + 1, threshold, state)
            if solved:
                return True, val
            if timeout_flag[0]:
                return False, math.inf
            if val < minimum:
                minimum = val
        return False, minimum

    threshold: float = heuristic(start)
    while True:
        if time.perf_counter() - t0 > time_limit:
            break
        solved, result = _dfs(start, 0, threshold, None)
        if solved:
            found = True
            solution_length = int(result)
            break
        if timeout_flag[0] or result == math.inf:
            break
        threshold = result

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return SolverResult(
        solved=found,
        solution_length=solution_length,
        nodes_expanded=nodes_expanded,
        peak_memory_mb=peak_bytes / (1024 * 1024),
        elapsed_seconds=time.perf_counter() - t0,
        algorithm=algorithm,
        heuristic_name=heuristic_name,
        weight=weight,
    )


def widastar(
    start: tuple[int, ...],
    heuristic: Callable[[tuple[int, ...]], int],
    weight: float = 1.5,
    heuristic_name: str = "h",
    time_limit: float = 300.0,
) -> SolverResult:
    """Convenience wrapper for Weighted IDA* with explicit weight."""
    return idastar(
        start,
        heuristic,
        heuristic_name=heuristic_name,
        weight=weight,
        time_limit=time_limit,
    )
