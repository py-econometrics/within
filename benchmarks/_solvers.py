"""Unified solve wrapper for benchmarks.

Provides ``run_solve()`` which calls ``solve()`` and returns a ``BenchmarkResult``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from within import solve

from ._types import BenchmarkResult, SolverConfig


def run_solve(
    categories: list[NDArray[np.int64]],
    n_levels: list[int],
    y: NDArray[np.float64],
    config: SolverConfig,
) -> BenchmarkResult:
    """Solve using *config* and return a timed result."""
    result = solve(
        np.column_stack(categories).astype(np.uintp),
        y,
        config.config,
        n_levels=n_levels,
    )

    return BenchmarkResult(
        problem="",  # Caller fills this in
        config=config.label,
        n_dofs=sum(n_levels),
        n_rows=len(categories[0]),
        setup_time=result.time_setup,
        solve_time=result.time_solve,
        iterations=result.iterations,
        final_residual=result.residual,
        converged=result.converged,
    )
