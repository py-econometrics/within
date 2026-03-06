"""Benchmark framework for domain-decomposition Cholesky solvers.

Public API
----------
- ``list_suites()`` — show registered benchmark suites
- ``run_suite(name, **kwargs)`` — run a suite by name
- ``BenchmarkResult`` — result dataclass
- ``SuiteOptions`` — options for suite runs
"""

from __future__ import annotations

from ._registry import SuiteInfo, SuiteOptions, get_suite, list_suites
from ._types import BenchmarkResult, ProblemSpec, SolverConfig

# Import all suite modules so their @suite decorators execute
import benchmarks.suites  # noqa: F401


def run_suite(
    name: str,
    seed: int = 42,
    tol: float = 1e-8,
    maxiter: int = 2000,
    quick: bool = False,
    filter_problems: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run a named benchmark suite and return results."""
    info = get_suite(name)
    opts = SuiteOptions(
        seed=seed,
        tol=tol,
        maxiter=maxiter,
        quick=quick,
        filter_problems=filter_problems,
    )
    return info.run_fn(opts)


__all__ = [
    "BenchmarkResult",
    "ProblemSpec",
    "SolverConfig",
    "SuiteInfo",
    "SuiteOptions",
    "get_suite",
    "list_suites",
    "run_suite",
]
