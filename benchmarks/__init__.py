"""Benchmark framework for domain-decomposition Cholesky solvers.

Public API
----------
- ``list_suites()`` — show registered benchmark suites
- ``BenchmarkResult`` — result dataclass
- ``SuiteOptions`` — options for suite runs
"""

from __future__ import annotations

from ._framework import (
    BenchmarkResult,
    ProblemSpec,
    SolverConfig,
    SuiteInfo,
    SuiteOptions,
    get_suite,
    list_suites,
)

# Import suite modules so their @suite decorators execute.
from .suites import (  # noqa: F401
    ac_comparison,
    akm_panel,
    fixest_comparison,
    high_fe,
    many_components,
    preconditioners,
    scaling,
    verify,
)


__all__ = [
    "BenchmarkResult",
    "ProblemSpec",
    "SolverConfig",
    "SuiteInfo",
    "SuiteOptions",
    "get_suite",
    "list_suites",
]
