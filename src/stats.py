"""
stats.py - Paired statistical tests for solver comparisons.

The 15-puzzle benchmark gives us *paired* observations: each solver runs
on the same set of instances, so we compare differences within instances
rather than between independent groups. The non-parametric paired
Wilcoxon signed-rank test is the standard choice here because solver
runtimes and node counts are heavily right-skewed.

Multiple comparisons across pairs of solvers inflate the family-wise
error rate, so we apply a Bonferroni correction by multiplying each
p-value by the number of comparisons (capped at 1.0). Bonferroni is
conservative but easy to interpret.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from scipy import stats


@dataclass
class PairedComparison:
    config_a: str
    config_b: str
    metric: str
    n_pairs: int                      # number of paired observations
    median_a: float
    median_b: float
    statistic: float                  # Wilcoxon W
    p_value: float                    # raw p-value
    p_corrected: float                # Bonferroni-corrected
    significant: bool                 # p_corrected < alpha


def paired_wilcoxon_table(
    results: list[dict],
    configs: Iterable[str],
    metric: str = "elapsed_seconds",
    only_solved_by_both: bool = True,
    alpha: float = 0.05,
) -> list[PairedComparison]:
    """Run a paired Wilcoxon signed-rank test on every pair of configs.

    Bonferroni correction multiplies the raw p-value by the number of
    pairs tested (capped at 1.0).

    Parameters
    ----------
    results : the JSON-style list of per-(config, instance) rows.
    configs : the configurations to compare.
    metric : name of the numeric column to compare ("elapsed_seconds",
             "nodes_expanded", or "peak_memory_mb" usually).
    only_solved_by_both : drop instances where either solver timed out;
                          comparing on TIMEOUT-coded values would be
                          dishonest because the algorithm never
                          actually finished.
    alpha : significance threshold for the *corrected* p-value.
    """
    configs_list = list(configs)
    n_pairs = len(configs_list) * (len(configs_list) - 1) // 2
    correction = max(n_pairs, 1)

    # Build per-config dict keyed on instance_idx
    by_cfg: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in results:
        by_cfg[r["config"]][r["instance_idx"]] = r

    out: list[PairedComparison] = []
    for a, b in combinations(configs_list, 2):
        rows_a = by_cfg.get(a, {})
        rows_b = by_cfg.get(b, {})
        common = sorted(set(rows_a) & set(rows_b))
        if only_solved_by_both:
            common = [
                i for i in common
                if rows_a[i]["solved"] and rows_b[i]["solved"]
            ]
        vals_a = [rows_a[i][metric] for i in common]
        vals_b = [rows_b[i][metric] for i in common]

        if len(vals_a) < 2 or vals_a == vals_b:
            # Wilcoxon undefined for n < 2 or all-zero differences.
            stat, raw_p = float("nan"), 1.0
        else:
            try:
                # 'pratt' includes zero differences in the ranking.
                test = stats.wilcoxon(vals_a, vals_b, zero_method="pratt")
                stat, raw_p = float(test.statistic), float(test.pvalue)
            except ValueError:
                stat, raw_p = float("nan"), 1.0

        p_corr = min(raw_p * correction, 1.0)
        med_a = stats.tmean(vals_a) if len(vals_a) > 0 else float("nan")
        med_b = stats.tmean(vals_b) if len(vals_b) > 0 else float("nan")

        out.append(
            PairedComparison(
                config_a=a,
                config_b=b,
                metric=metric,
                n_pairs=len(common),
                median_a=float(_median(vals_a)),
                median_b=float(_median(vals_b)),
                statistic=stat,
                p_value=raw_p,
                p_corrected=p_corr,
                significant=p_corr < alpha,
            )
        )
    return out


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def format_comparison_table(
    comparisons: list[PairedComparison],
) -> str:
    """Format a list of PairedComparison objects as a fixed-width table."""
    rows = [
        f"{'Config A':<28} {'Config B':<28} {'n':>4} "
        f"{'med A':>10} {'med B':>10} {'W':>10} {'p_corr':>10} sig"
    ]
    rows.append("-" * len(rows[0]))
    for c in comparisons:
        sig_marker = "***" if c.significant else ""
        rows.append(
            f"{c.config_a:<28} {c.config_b:<28} {c.n_pairs:>4} "
            f"{c.median_a:>10.3f} {c.median_b:>10.3f} "
            f"{c.statistic:>10.1f} {c.p_corrected:>10.4g} {sig_marker}"
        )
    return "\n".join(rows)
