from __future__ import annotations

import statistics
from dataclasses import dataclass

from ._framework import BenchmarkResult


@dataclass(frozen=True)
class Comparison:
    suite: str
    problem: str
    config: str
    full: BenchmarkResult | None
    reduced: BenchmarkResult | None


def print_comparison(rows: list[Comparison]) -> None:
    print("\nFull / reduced total-time ratios (>1 favors reduced)")
    print(
        f"{'Suite':30} {'Pairs':>6} {'Median':>8} {'Full s':>10} {'Reduced s':>10} {'Converged F/R':>15}"
    )
    for suite in sorted({r.suite for r in rows}):
        pairs = [r for r in rows if r.suite == suite and r.full and r.reduced]
        full = [r.full.setup_time + r.full.solve_time for r in pairs]
        reduced = [r.reduced.setup_time + r.reduced.solve_time for r in pairs]
        ratios = [f / r for f, r in zip(full, reduced) if r > 0]
        converged = [
            sum(r.full.converged for r in pairs),
            sum(r.reduced.converged for r in pairs),
        ]
        median = statistics.median(ratios) if ratios else float("nan")
        print(
            f"{suite:30} {len(pairs):6} {median:8.2f} {sum(full):10.3f} {sum(reduced):10.3f} {converged[0]:6}/{converged[1]:<6}"
        )
    missing = sum(r.full is None or r.reduced is None for r in rows)
    print(f"Unpaired results: {missing}")
