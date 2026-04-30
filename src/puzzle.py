"""
puzzle.py - 15-Puzzle state representation and core utilities.

A state is a tuple of 16 integers in row-major order (index 0..15),
where 0 represents the blank tile.  Goal state: (1, 2, ..., 15, 0).

This module provides:
  * Solvability checking via inversion parity.
  * Successor generation (slide blank up/down/left/right).
  * Manhattan distance and Linear Conflict heuristics (admissible).
  * Incremental Manhattan-distance delta for fast IDA* re-evaluation.
  * 64-bit compact state encoding for hashing and PDB lookups.
  * Pretty-printer for debugging.
"""

from __future__ import annotations
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOAL_STATE: tuple[int, ...] = tuple(range(1, 16)) + (0,)

#: Goal (row, col) for each tile value (0 included for completeness).
GOAL_POS: dict[int, tuple[int, int]] = {
    tile: (idx // 4, idx % 4)
    for idx, tile in enumerate(GOAL_STATE)
}

#: Goal flat index for each tile value.
GOAL_FLAT: dict[int, int] = {
    tile: idx for idx, tile in enumerate(GOAL_STATE)
}

# Pre-computed adjacency: NEIGHBOURS[i] = tuple of flat indices reachable from i
NEIGHBOURS: list[tuple[int, ...]] = []
for _i in range(16):
    _r, _c = _i // 4, _i % 4
    _n: list[int] = []
    if _r > 0: _n.append(_i - 4)
    if _r < 3: _n.append(_i + 4)
    if _c > 0: _n.append(_i - 1)
    if _c < 3: _n.append(_i + 1)
    NEIGHBOURS.append(tuple(_n))


# ---------------------------------------------------------------------------
# Solvability
# ---------------------------------------------------------------------------

def is_solvable(state: tuple[int, ...]) -> bool:
    """Return True iff *state* is reachable from GOAL_STATE.

    A 4x4 sliding-tile configuration is solvable iff
        inversions + blank_row_from_bottom  is even,
    where blank_row_from_bottom counts from 0 at the bottom row.
    """
    tiles = [t for t in state if t != 0]
    inversions = sum(
        1
        for i in range(len(tiles))
        for j in range(i + 1, len(tiles))
        if tiles[i] > tiles[j]
    )
    blank_idx = state.index(0)
    blank_row_from_bottom = 3 - (blank_idx // 4)
    return (inversions + blank_row_from_bottom) % 2 == 0


# ---------------------------------------------------------------------------
# Successor generation
# ---------------------------------------------------------------------------

def get_successors(state: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield (successor_state, move_cost=1) for each legal slide."""
    blank = state.index(0)
    lst = list(state)
    for nb in NEIGHBOURS[blank]:
        lst[blank], lst[nb] = lst[nb], lst[blank]
        yield tuple(lst), 1
        lst[blank], lst[nb] = lst[nb], lst[blank]


def get_successors_full(
    state: tuple[int, ...],
) -> Iterator[tuple[tuple[int, ...], int, int, int]]:
    """Yield (successor_state, blank_dest, moved_tile, blank_src) for each slide.

    Used by IDA* for the no-reversal pruning and for incremental Manhattan
    updates without recomputing the full heuristic each step.
    """
    blank = state.index(0)
    lst = list(state)
    for nb in NEIGHBOURS[blank]:
        moved_tile = state[nb]
        lst[blank], lst[nb] = lst[nb], lst[blank]
        yield tuple(lst), nb, moved_tile, blank
        lst[blank], lst[nb] = lst[nb], lst[blank]


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def manhattan_distance(state: tuple[int, ...]) -> int:
    """Sum of Manhattan distances of all tiles from their goal positions.

    Admissible and consistent. O(16) per call.
    """
    total = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        gr, gc = GOAL_POS[tile]
        sr, sc = idx >> 2, idx & 3
        total += abs(gr - sr) + abs(gc - sc)
    return total


def manhattan_delta(moved_tile: int, from_idx: int, to_idx: int) -> int:
    """Change in Manhattan distance when *moved_tile* slides from from_idx
    to to_idx. Lets IDA* update h in O(1) per step instead of O(16).
    """
    if moved_tile == 0:
        return 0
    gr, gc = GOAL_POS[moved_tile]
    fr, fc = from_idx >> 2, from_idx & 3
    tr, tc = to_idx >> 2, to_idx & 3
    new_d = abs(gr - tr) + abs(gc - tc)
    old_d = abs(gr - fr) + abs(gc - fc)
    return new_d - old_d


def linear_conflict(state: tuple[int, ...]) -> int:
    """Manhattan distance + linear-conflict correction.

    For each row/column, if two tiles that both belong in that row/column
    are in reversed order they must pass each other, costing at least 2
    extra moves per conflicting pair. The combined heuristic dominates
    Manhattan distance and remains admissible and consistent.
    """
    md = manhattan_distance(state)
    lc = 0

    # Row conflicts
    for r in range(4):
        row_tiles: list[tuple[int, int]] = []
        for c in range(4):
            tile = state[r * 4 + c]
            if tile != 0 and GOAL_POS[tile][0] == r:
                row_tiles.append((GOAL_POS[tile][1], c))  # (goal_col, cur_col)
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                gci, ci = row_tiles[i]
                gcj, cj = row_tiles[j]
                if (gci > gcj) != (ci > cj):
                    lc += 2

    # Column conflicts
    for c in range(4):
        col_tiles: list[tuple[int, int]] = []
        for r in range(4):
            tile = state[r * 4 + c]
            if tile != 0 and GOAL_POS[tile][1] == c:
                col_tiles.append((GOAL_POS[tile][0], r))
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                gri, ri = col_tiles[i]
                grj, rj = col_tiles[j]
                if (gri > grj) != (ri > rj):
                    lc += 2

    return md + lc


# ---------------------------------------------------------------------------
# Encoding / printing
# ---------------------------------------------------------------------------

def encode_state(state: tuple[int, ...]) -> int:
    """Pack state into a 64-bit int (4 bits per tile, MSB-first)."""
    val = 0
    for tile in state:
        val = (val << 4) | tile
    return val


def decode_state(code: int) -> tuple[int, ...]:
    """Inverse of encode_state."""
    out: list[int] = []
    for _ in range(16):
        out.append(code & 0xF)
        code >>= 4
    return tuple(reversed(out))


def pretty_print(state: tuple[int, ...]) -> str:
    """Return a 4x4 grid string for display."""
    rows = []
    for r in range(4):
        row = []
        for c in range(4):
            t = state[r * 4 + c]
            row.append(f"{t:2d}" if t != 0 else " _")
        rows.append(" ".join(row))
    return "\n".join(rows)
