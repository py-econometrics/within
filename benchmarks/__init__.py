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

# Import suite modules so their @suite decorators execute.
from .suites import (  # noqa: F401
    ac_comparison,
    akm_panel,
    fixest_comparison,
    graph_backend,
    high_fe,
    iteration_reduction,
    laplacian_2fe,
    many_components,
    one_level_baseline,
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
