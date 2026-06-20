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
        LocalSolverConfig,
        ReductionStrategy,
    )

    schwarz = AdditiveSchwarz(
        local_solver=LocalSolverConfig(
            approx_chol=ApproxCholConfig(split_merge=2),
        ),
        reduction=ReductionStrategy.Auto,
    )
    result = solve(categories, y, preconditioner=schwarz)
"""

from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    LocalSolverConfig as _NativeLocalSolverConfig,
    ReductionStrategy,
)

# Module-level constant capturing the library-default approximate Schur config.
# Used as the kwarg default for `LocalSolverConfig` so that omitting the
# argument means "library default" while explicitly passing ``approx_schur=None``
# means "exact Schur" — preserving the Rust ``Option<ApproxSchurConfig>``
# semantics where ``None`` is exact.
_DEFAULT_APPROX_SCHUR = ApproxSchurConfig()


class LocalSolverConfig(_NativeLocalSolverConfig):
    """Local solver configuration for Schwarz subdomains (Schur reduction).

    ``approx_schur`` carries three-way semantics:

    - Omitted: use the library default (approximate Schur with clique-tree
      sampling).
    - ``None``: request an exact Schur complement (slower per subdomain, used
      for validation benchmarks).
    - An ``ApproxSchurConfig(...)`` instance: approximate Schur with custom
      seed/split.
    """

    __slots__ = ()  # Mirror the native class's `frozen` semantics — no extra attrs.

    def __new__(
        cls,
        approx_chol: ApproxCholConfig | None = None,
        approx_schur: ApproxSchurConfig | None = _DEFAULT_APPROX_SCHUR,
        dense_threshold: int | None = None,
    ) -> "LocalSolverConfig":
        return _NativeLocalSolverConfig.__new__(
            cls,
            approx_chol=approx_chol,
            approx_schur=approx_schur,
            dense_threshold=dense_threshold,
        )

    def __reduce__(self):
        # Pickle through this Python wrapper (not the native class) so unpickle
        # goes back through the default-injection logic in ``__new__``.
        return (
            LocalSolverConfig,
            (self.approx_chol, self.approx_schur, self.dense_threshold),
        )


__all__ = [
    "AdditiveSchwarz",
    "ApproxCholConfig",
    "ApproxSchurConfig",
    "LocalSolverConfig",
    "ReductionStrategy",
]
