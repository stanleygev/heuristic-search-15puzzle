# Heuristic Search in the 15-Puzzle

CS 57200: Heuristic Problem Solving — Stanley Gevers

Implementation and experimental comparison of heuristic search algorithms for optimally solving random 15-puzzle instances, including A\*, IDA\*, disjoint pattern database heuristics, and weighted A\*.

---

## Requirements

- Python 3.10 or higher
- pip

---

## Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Code

All commands are run from the `src/` directory:

```bash
cd src
```

### Run test cases

Verifies correctness of the baseline and linear-conflict heuristic on 5 known instances. No PDB required — runs in a few seconds.

```bash
python main.py --mode test
```

### Run a quick benchmark

Builds the 5-5-5 PDB (cached after first run) and benchmarks 3 algorithms on a small instance set.

```bash
python main.py --mode quick --n 10
```

### Run the full benchmark

Runs all 8 solver configurations on 100 instances and generates result figures.

```bash
python main.py --mode benchmark --n 100
```

Results are saved to `results/results.json`. Figures are saved to `results/figures/`.

---

## Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `test` | `test`, `quick`, or `benchmark` |
| `--n` | `100` | Number of benchmark instances |
| `--partition` | `5-5-5` | PDB tile partition: `5-5-5`, `6-5-4`, or `7-8` |
| `--time-limit` | `60` | Per-instance timeout (seconds) |
| `--seed` | `42` | Random seed for instance generation |
| `--cache` | `data/pdb_cache` | Directory for cached PDB files |
| `--results` | `results/results.json` | Output path for raw results |
| `--figures` | `results/figures` | Output directory for figures |

---

## Project Structure

```
fifteen_puzzle/
├── src/
│   ├── puzzle.py        State representation, heuristics, successor generation
│   ├── pdb.py           Pattern database construction and lookup
│   ├── solvers.py       A*, IDA*, Weighted A*
│   ├── benchmark.py     Instance generation and experimental runner
│   ├── analysis.py      Statistics and matplotlib figures
│   └── main.py          CLI entry point
├── data/
│   ├── benchmark.json   100 fixed instances (seed=42)
│   └── pdb_cache/       Serialized PDB tables (generated on first run)
├── results/
│   ├── results.json     Raw benchmark output
│   └── figures/         Generated PNG figures
└── requirements.txt
```

---

## Algorithms Implemented

- **A\* with Manhattan Distance** — admissible baseline
- **A\* with Linear Conflict** — augments Manhattan distance with row/column conflict correction
- **A\* with Disjoint PDB** — additive pattern database heuristic (5-5-5, 6-5-4, or 7-8 partition)
- **IDA\* with Disjoint PDB** — optimal solver with linear memory usage
- **Weighted A\*** — bounded-suboptimal solver with weights w ∈ {1.25, 1.5, 2.0}

---

## Reproducibility

The benchmark suite is fixed at seed 42 and stored in `data/benchmark.json`. PDB tables are cached to `data/pdb_cache/` after first construction. All results can be reproduced exactly by running `python main.py --mode benchmark` on any machine.
