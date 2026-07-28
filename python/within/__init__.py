"""High-performance fixed-effects solver for econometric panel data."""

from within._within import (
    BatchSolveResult,
    CoefficientLayout,
    Effect,
    LsmrOptions,
    Preconditioner,
    PreconditionerConfig,
    SolveResult,
    Solver,
    UnidentifiedDirection,
    solve,
    solve_batch,
)
from within import config  # noqa: F401 — expose submodule on `within.config`

__all__ = [
    "BatchSolveResult",
    "CoefficientLayout",
    "Effect",
    "LsmrOptions",
    "Preconditioner",
    "PreconditionerConfig",
    "SolveResult",
    "Solver",
    "UnidentifiedDirection",
    "solve",
    "solve_batch",
    "config",
]
