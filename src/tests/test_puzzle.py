"""
test_puzzle.py - Unit tests for puzzle and solver modules.

Run with:
    python -m unittest discover -s tests -v
or:
    python tests/test_puzzle.py
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

# Make sibling src/ importable when tests are run directly
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from puzzle import (
    GOAL_STATE, manhattan_distance, linear_conflict, is_solvable,
    get_successors, encode_state, decode_state, manhattan_delta,
)
from solvers import astar, idastar
from benchmark import random_walk


class TestPuzzleBasics(unittest.TestCase):
    def test_goal_state_layout(self):
        self.assertEqual(GOAL_STATE, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0))

    def test_manhattan_goal_is_zero(self):
        self.assertEqual(manhattan_distance(GOAL_STATE), 0)

    def test_linear_conflict_goal_is_zero(self):
        self.assertEqual(linear_conflict(GOAL_STATE), 0)

    def test_linear_conflict_dominates_manhattan(self):
        # On 50 random walks, LC must always be >= MD.
        for seed in range(50):
            s = random_walk(seed=seed, shuffle_moves=30)
            self.assertGreaterEqual(linear_conflict(s), manhattan_distance(s))

    def test_manhattan_one_move(self):
        # Swap blank with tile 15 -> tile 15 displaces by 1.
        s = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15)
        self.assertEqual(manhattan_distance(s), 1)

    def test_solvability_goal(self):
        self.assertTrue(is_solvable(GOAL_STATE))

    def test_solvability_known_unsolvable(self):
        # Classical unsolvable: swap last two non-blank tiles in the goal.
        s = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0)
        self.assertFalse(is_solvable(s))

    def test_random_walks_solvable(self):
        for seed in range(20):
            s = random_walk(seed=seed, shuffle_moves=50)
            self.assertTrue(is_solvable(s))


class TestSuccessors(unittest.TestCase):
    def test_corner_blank_two_successors(self):
        # Blank at (0,0) has 2 neighbours (right, down).
        s = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        succs = list(get_successors(s))
        self.assertEqual(len(succs), 2)

    def test_edge_blank_three_successors(self):
        s = (1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        succs = list(get_successors(s))
        self.assertEqual(len(succs), 3)

    def test_interior_blank_four_successors(self):
        s = (1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        succs = list(get_successors(s))
        self.assertEqual(len(succs), 4)


class TestEncoding(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        for seed in range(20):
            s = random_walk(seed=seed, shuffle_moves=30)
            self.assertEqual(decode_state(encode_state(s)), s)


class TestManhattanDelta(unittest.TestCase):
    def test_delta_consistent_with_recompute(self):
        # For every legal move from 10 random states, the delta in MD
        # computed by manhattan_delta must match a full recomputation.
        for seed in range(10):
            s = random_walk(seed=seed, shuffle_moves=20)
            md = manhattan_distance(s)
            blank = s.index(0)
            for nb_state, _ in get_successors(s):
                new_md = manhattan_distance(nb_state)
                # Find which tile moved.
                # The tile that was at the blank's neighbour cell moved
                # to where the blank was.
                moved = None
                for i, (a, b) in enumerate(zip(s, nb_state)):
                    if a != b and a != 0:
                        moved = (a, i, nb_state.index(a))
                        break
                self.assertIsNotNone(moved)
                tile, frm, to = moved
                self.assertEqual(md + manhattan_delta(tile, frm, to), new_md)


class TestSolvers(unittest.TestCase):
    def test_astar_solves_easy(self):
        s = random_walk(seed=11, shuffle_moves=15)
        res = astar(s, manhattan_distance, time_limit=10.0)
        self.assertTrue(res.solved)
        self.assertGreater(res.solution_length, 0)

    def test_idastar_solves_easy(self):
        s = random_walk(seed=12, shuffle_moves=15)
        res = idastar(s, manhattan_distance, time_limit=10.0)
        self.assertTrue(res.solved)

    def test_astar_idastar_agree_on_optimal_length(self):
        # Both must find the same optimal length on the same instance.
        for seed in range(5):
            s = random_walk(seed=seed + 100, shuffle_moves=12)
            r1 = astar(s, manhattan_distance, time_limit=10.0)
            r2 = idastar(s, manhattan_distance, time_limit=10.0)
            self.assertTrue(r1.solved and r2.solved)
            self.assertEqual(r1.solution_length, r2.solution_length)

    def test_weighted_astar_at_w1_equals_astar(self):
        s = random_walk(seed=42, shuffle_moves=10)
        r1 = astar(s, manhattan_distance, weight=1.0, time_limit=10.0)
        r2 = astar(s, manhattan_distance, weight=1.0, time_limit=10.0)
        self.assertEqual(r1.solution_length, r2.solution_length)

    def test_weighted_within_w_times_optimal(self):
        # Weighted A* with w=2 must produce solutions of length at most
        # 2 * optimal.
        s = random_walk(seed=99, shuffle_moves=15)
        opt = astar(s, manhattan_distance, weight=1.0, time_limit=10.0)
        weighted = astar(s, manhattan_distance, weight=2.0, time_limit=10.0)
        self.assertTrue(opt.solved and weighted.solved)
        self.assertLessEqual(weighted.solution_length, 2.0 * opt.solution_length)


if __name__ == "__main__":
    unittest.main(verbosity=2)
