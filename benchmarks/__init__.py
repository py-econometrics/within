"""Benchmark framework for domain-decomposition Cholesky solvers.

Public API
----------
- ``list_suites()`` — show registered benchmark suites
- ``BenchmarkResult`` — result dataclass
- ``SuiteOptions`` — options for suite runs
"""

from __future__ import annotations

from ._registry import SuiteInfo, SuiteOptions, get_suite, list_suites
from ._types import BenchmarkResult, ProblemSpec, SolverConfig

# Import all suite modules so their @suite decorators execute
import benchmarks.suites  # noqa: F401


__all__ = [
    "BenchmarkResult",
    "ProblemSpec",
    "SolverConfig",
    "SuiteInfo",
    "SuiteOptions",
    "get_suite",
    "list_suites",
]
