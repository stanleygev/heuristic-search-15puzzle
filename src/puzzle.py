"""
puzzle.py - 15-Puzzle state representation and core utilities.

The state is a tuple of 16 integers in row-major order (index 0..15),
where 0 represents the blank tile.  Goal state: (1,2,...,15,0).
"""

from __future__ import annotations
from typing import Optional, Iterator

GOAL_STATE: tuple[int, ...] = tuple(range(1, 16)) + (0,)

# Pre-compute goal positions for fast Manhattan distance computation
GOAL_POS: dict[int, tuple[int, int]] = {
    tile: (tile_idx // 4, tile_idx % 4)
    for tile_idx, tile in enumerate(GOAL_STATE)
}

# Neighbours: maps flat index -> list of reachable flat indices
_NEIGHBOURS: list[list[int]] = []
for _i in range(16):
    _r, _c = _i // 4, _i % 4
    _n: list[int] = []
    if _r > 0: _n.append(_i - 4)
    if _r < 3: _n.append(_i + 4)
    if _c > 0: _n.append(_i - 1)
    if _c < 3: _n.append(_i + 1)
    _NEIGHBOURS.append(_n)


def is_solvable(state: tuple[int, ...]) -> bool:
    """Return True iff *state* is reachable from GOAL_STATE.

    A configuration is solvable iff
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


def get_successors(state: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield (successor_state, move_cost) for each legal slide."""
    blank = state.index(0)
    lst = list(state)
    for nb in _NEIGHBOURS[blank]:
        lst[blank], lst[nb] = lst[nb], lst[blank]
        yield tuple(lst), 1
        lst[blank], lst[nb] = lst[nb], lst[blank]


def manhattan_distance(state: tuple[int, ...]) -> int:
    """Sum of Manhattan distances of all tiles from their goal positions."""
    total = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        gr, gc = GOAL_POS[tile]
        sr, sc = idx // 4, idx % 4
        total += abs(gr - sr) + abs(gc - sc)
    return total


def linear_conflict(state: tuple[int, ...]) -> int:
    """Manhattan distance + linear-conflict correction.

    For each row/column, if two tiles that both belong in that row/column
    are in reversed order, they must pass each other, costing at least 2
    extra moves per conflicting pair.
    """
    md = manhattan_distance(state)
    lc = 0

    # Row conflicts
    for r in range(4):
        row_tiles = []
        for c in range(4):
            tile = state[r * 4 + c]
            if tile != 0 and GOAL_POS[tile][0] == r:
                row_tiles.append((GOAL_POS[tile][1], c))  # (goal_col, cur_col)
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                # If tile i is to the right of tile j in goal, but left currently
                # — they are in conflict
                if row_tiles[i][0] > row_tiles[j][0] and row_tiles[i][1] < row_tiles[j][1]:
                    lc += 2
                elif row_tiles[i][0] < row_tiles[j][0] and row_tiles[i][1] > row_tiles[j][1]:
                    lc += 2

    # Column conflicts
    for c in range(4):
        col_tiles = []
        for r in range(4):
            tile = state[r * 4 + c]
            if tile != 0 and GOAL_POS[tile][1] == c:
                col_tiles.append((GOAL_POS[tile][0], r))  # (goal_row, cur_row)
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                if col_tiles[i][0] > col_tiles[j][0] and col_tiles[i][1] < col_tiles[j][1]:
                    lc += 2
                elif col_tiles[i][0] < col_tiles[j][0] and col_tiles[i][1] > col_tiles[j][1]:
                    lc += 2

    return md + lc


def encode_state(state: tuple[int, ...]) -> int:
    """Pack state into a 64-bit int (4 bits per tile)."""
    val = 0
    for tile in state:
        val = (val << 4) | tile
    return val


def pretty_print(state: tuple[int, ...]) -> str:
    """Return a 4×4 grid string for display."""
    rows = []
    for r in range(4):
        row = []
        for c in range(4):
            t = state[r * 4 + c]
            row.append(f"{t:2d}" if t != 0 else " _")
        rows.append(" ".join(row))
    return "\n".join(rows)
