"""Type stubs for the Rust extension module."""

import numpy as np
from numpy.typing import NDArray
from enum import IntEnum

class OperatorRepr(IntEnum):
    Implicit = 0
    Explicit = 1

class Preconditioner(IntEnum):
    Additive = 0
    Multiplicative = 1
    Off = 2

class CG:
    """Conjugate Gradient solver configuration.

    ``preconditioner`` accepts:
    - ``None`` (default) — additive Schwarz with default local solver
    - ``Preconditioner.Additive`` — same as ``None`` (explicit)
    - ``Preconditioner.Off`` — unpreconditioned CG
    - ``AdditiveSchwarz(...)`` — advanced fine-grained config (import from ``within._within``)
    """

    tol: float
    maxiter: int
    preconditioner: Preconditioner | AdditiveSchwarz | None
    operator: OperatorRepr
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        preconditioner: Preconditioner | AdditiveSchwarz | None = None,
        operator: OperatorRepr = OperatorRepr.Implicit,
    ) -> None: ...

class GMRES:
    """GMRES solver configuration.

    ``preconditioner`` accepts:
    - ``None`` (default) — additive Schwarz with default local solver
    - ``Preconditioner.Additive`` — same as ``None`` (explicit)
    - ``Preconditioner.Multiplicative`` — multiplicative Schwarz with default local solver
    - ``Preconditioner.Off`` — unpreconditioned GMRES
    - ``AdditiveSchwarz(...)`` / ``MultiplicativeSchwarz(...)`` — advanced config
    """

    tol: float
    maxiter: int
    restart: int
    preconditioner: Preconditioner | AdditiveSchwarz | MultiplicativeSchwarz | None
    operator: OperatorRepr
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        restart: int = 30,
        preconditioner: Preconditioner
        | AdditiveSchwarz
        | MultiplicativeSchwarz
        | None = None,
        operator: OperatorRepr = OperatorRepr.Implicit,
    ) -> None: ...

class SolveResult:
    """Result of a solve operation."""

    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def time_total(self) -> float: ...
    @property
    def time_setup(self) -> float: ...
    @property
    def time_solve(self) -> float: ...

def solve(
    categories: NDArray[np.uint32],
    y: NDArray[np.float64],
    config: CG | GMRES | None = None,
    weights: NDArray[np.float64] | None = None,
) -> SolveResult:
    """Solve fixed-effects normal equations.

    ``categories`` should be F-contiguous (column-major) for best performance.
    Use ``np.asfortranarray(categories)`` to convert.  A ``UserWarning`` is
    emitted when a C-contiguous array is passed.
    """
    ...

# ---------------------------------------------------------------------------
# Advanced config classes (for benchmarks / power users)
# ---------------------------------------------------------------------------

class ApproxCholConfig:
    seed: int
    split: int
    def __init__(self, seed: int = 0, split: int = 1) -> None: ...

class ApproxSchurConfig:
    seed: int
    split: int
    def __init__(self, seed: int = 0, split: int = 1) -> None: ...

class SchurComplement:
    approx_chol: ApproxCholConfig | None
    approx_schur: ApproxSchurConfig | None
    dense_threshold: int
    def __init__(
        self,
        approx_chol: ApproxCholConfig | None = None,
        approx_schur: ApproxSchurConfig | None = None,
        dense_threshold: int = 24,
    ) -> None: ...

class FullSddm:
    approx_chol: ApproxCholConfig | None
    def __init__(self, approx_chol: ApproxCholConfig | None = None) -> None: ...

class AdditiveSchwarz:
    local_solver: SchurComplement | FullSddm | None
    def __init__(
        self,
        local_solver: SchurComplement | FullSddm | None = None,
    ) -> None: ...

class MultiplicativeSchwarz:
    local_solver: SchurComplement | FullSddm | None
    def __init__(
        self,
        local_solver: SchurComplement | FullSddm | None = None,
    ) -> None: ...
