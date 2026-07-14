"""Advanced configuration objects for :mod:`within`.

The :mod:`within` top-level namespace exposes only the call-site essentials
(``solve``, ``Solver``, ``LsmrOptions``, ``PreconditionerConfig``,
``Preconditioner``). The advanced configuration types used to fine-tune the
preconditioner live in this submodule.

Typical usage::

    from within import solve, LsmrOptions
    from within.config import (
        AdditiveSchwarz,
        ApproxCholConfig,
        ApproxSchurConfig,
        LocalSolverConfig,
        ReductionStrategy,
        Schur,
    )

    schwarz = AdditiveSchwarz(
        local_solver=LocalSolverConfig(
            approx_chol=ApproxCholConfig(split_merge=2),
            schur=Schur.approximate(ApproxSchurConfig(split=2)),
        ),
        reduction=ReductionStrategy.Auto,
    )
    result = solve(categories, y, preconditioner=schwarz)

``LocalSolverConfig`` omits ``schur`` for the library default (approximate
Schur); pass ``Schur.exact()`` for the exact complement.
"""

from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    LocalSolverConfig,
    ReductionStrategy,
    ScalingConfig,
    Schur,
)

__all__ = [
    "AdditiveSchwarz",
    "ApproxCholConfig",
    "ApproxSchurConfig",
    "LocalSolverConfig",
    "ReductionStrategy",
    "ScalingConfig",
    "Schur",
]
