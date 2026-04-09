"""
pdb.py - Disjoint Pattern Database (PDB) heuristics for the 15-puzzle.

A PDB stores, for a subset of tiles, the exact number of moves required
to place those tiles in their goal positions (ignoring all other tiles).
Multiple non-overlapping PDBs can be combined additively (disjoint PDBs)
to produce an admissible, highly informative heuristic.

Supported partition strategies:
  * 5-5-5  partition  (3 tables, each holding 5 tiles)
  * 6-5-4  partition  (3 tables)
  * 7-8    partition  (2 tables)  ← default for best heuristic quality
"""

from __future__ import annotations
import collections
import os
import pickle
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Default partition strategies
# ---------------------------------------------------------------------------
PARTITION_7_8 = (
    (1, 2, 3, 4, 5, 6, 7),
    (8, 9, 10, 11, 12, 13, 14, 15),
)
PARTITION_5_5_5 = (
    (1, 2, 3, 4, 5),
    (6, 7, 8, 9, 10),
    (11, 12, 13, 14, 15),
)
PARTITION_6_5_4 = (
    (1, 2, 3, 4, 5, 6),
    (7, 8, 9, 10, 11),
    (12, 13, 14, 15),
)

# ---------------------------------------------------------------------------
# Goal information
# ---------------------------------------------------------------------------
_GOAL = tuple(range(1, 16)) + (0,)
_GOAL_POS: dict[int, int] = {tile: idx for idx, tile in enumerate(_GOAL)}

# Adjacency list (flat-index based)
_NEIGHBOURS: list[list[int]] = []
for _i in range(16):
    _r, _c = _i // 4, _i % 4
    _n: list[int] = []
    if _r > 0: _n.append(_i - 4)
    if _r < 3: _n.append(_i + 4)
    if _c > 0: _n.append(_i - 1)
    if _c < 3: _n.append(_i + 1)
    _NEIGHBOURS.append(_n)


# ---------------------------------------------------------------------------
# Single PDB
# ---------------------------------------------------------------------------

class PatternDatabase:
    """Pre-computed exact distances for a tile subset.

    The database maps a *pattern* (positions of the tracked tiles and
    the blank) to the minimum number of moves to place those tiles at
    their goal positions.  The blank is always tracked so the BFS can
    generate successors.
    """

    def __init__(self, tiles: Sequence[int]) -> None:
        self.tiles: tuple[int, ...] = tuple(sorted(tiles))
        self._table: dict[tuple[int, ...], int] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, verbose: bool = False) -> None:
        """Backward BFS from the goal pattern to fill the table."""
        # Pattern = tuple of length len(tiles)+1 holding
        #   (blank_pos, tile1_pos, tile2_pos, ...)
        # in a canonical order: blank first, then tiles sorted by tile value.

        goal_pattern = self._state_to_pattern(_GOAL)
        queue: collections.deque[tuple[tuple[int, ...], int]] = collections.deque()
        queue.append((goal_pattern, 0))
        self._table[goal_pattern] = 0

        if verbose:
            print(f"  Building PDB for tiles {self.tiles} …", flush=True)

        while queue:
            pattern, dist = queue.popleft()
            blank_pos = pattern[0]

            for nb in _NEIGHBOURS[blank_pos]:
                # Determine whether the neighbour cell holds a tracked tile
                nb_tile_idx = self._find_tile_at(pattern, nb)

                if nb_tile_idx is not None:
                    # Moving a tracked tile counts as a move
                    new_pattern = list(pattern)
                    new_pattern[0] = nb                    # blank moves to nb
                    new_pattern[nb_tile_idx] = blank_pos   # tile moves to blank_pos
                    new_pattern_t = tuple(new_pattern)
                    if new_pattern_t not in self._table:
                        self._table[new_pattern_t] = dist + 1
                        queue.append((new_pattern_t, dist + 1))
                else:
                    # Moving a non-tracked tile: blank moves, no distance cost
                    new_pattern = list(pattern)
                    new_pattern[0] = nb
                    new_pattern_t = tuple(new_pattern)
                    if new_pattern_t not in self._table:
                        self._table[new_pattern_t] = dist
                        queue.append((new_pattern_t, dist))

        if verbose:
            print(f"  PDB for tiles {self.tiles}: {len(self._table):,} entries", flush=True)

    def _state_to_pattern(self, state: tuple[int, ...]) -> tuple[int, ...]:
        """Extract the pattern (blank_pos, tile1_pos, ...) from a full state."""
        pos = [0] * (len(self.tiles) + 1)
        pos[0] = state.index(0)
        for k, tile in enumerate(self.tiles):
            pos[k + 1] = state.index(tile)
        return tuple(pos)

    def _find_tile_at(self, pattern: tuple[int, ...], flat_idx: int) -> int | None:
        """Return the 1-based index into *pattern* of the tile at *flat_idx*, or None."""
        for k in range(1, len(pattern)):
            if pattern[k] == flat_idx:
                return k
        return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, state: tuple[int, ...]) -> int:
        pattern = self._state_to_pattern(state)
        return self._table.get(pattern, 0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump((self.tiles, self._table), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PatternDatabase":
        with open(path, "rb") as f:
            tiles, table = pickle.load(f)
        pdb = cls(tiles)
        pdb._table = table
        return pdb

    def __len__(self) -> int:
        return len(self._table)


# ---------------------------------------------------------------------------
# Disjoint PDB heuristic
# ---------------------------------------------------------------------------

class DisjointPDB:
    """Additive combination of multiple non-overlapping PDBs.

    Parameters
    ----------
    partition :
        Sequence of tile-group sequences. Together they must cover
        exactly tiles 1-15 without overlap.
    cache_dir :
        Directory to persist / reload built databases.
    """

    def __init__(
        self,
        partition: Sequence[Sequence[int]] = PARTITION_7_8,
        cache_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self._pdbs: list[PatternDatabase] = []
        cache_dir_path = Path(cache_dir) if cache_dir else None

        for group in partition:
            pdb = PatternDatabase(group)
            cached_path = (
                cache_dir_path / f"pdb_{'_'.join(str(t) for t in sorted(group))}.pkl"
                if cache_dir_path
                else None
            )
            if cached_path and cached_path.exists():
                if verbose:
                    print(f"  Loading cached PDB from {cached_path} …", flush=True)
                pdb = PatternDatabase.load(cached_path)
            else:
                pdb.build(verbose=verbose)
                if cached_path:
                    cache_dir_path.mkdir(parents=True, exist_ok=True)
                    pdb.save(cached_path)
                    if verbose:
                        print(f"  Saved PDB to {cached_path}", flush=True)
            self._pdbs.append(pdb)

    def __call__(self, state: tuple[int, ...]) -> int:
        """Return the additive PDB heuristic value for *state*."""
        return sum(pdb.lookup(state) for pdb in self._pdbs)
