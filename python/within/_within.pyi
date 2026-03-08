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
    """Conjugate Gradient solver configuration."""

    tol: float
    maxiter: int
    operator: OperatorRepr
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        operator: OperatorRepr = OperatorRepr.Implicit,
    ) -> None: ...

class GMRES:
    """GMRES solver configuration."""

    tol: float
    maxiter: int
    restart: int
    operator: OperatorRepr
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        restart: int = 30,
        operator: OperatorRepr = OperatorRepr.Implicit,
    ) -> None: ...

class SolveResult:
    """Result of a solve operation."""

    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def demeaned(self) -> NDArray[np.float64]: ...
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

class BatchSolveResult:
    """Result of a batch solve across multiple RHS vectors."""

    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def demeaned(self) -> NDArray[np.float64]: ...
    @property
    def converged(self) -> list[bool]: ...
    @property
    def iterations(self) -> list[int]: ...
    @property
    def residual(self) -> list[float]: ...
    @property
    def time_solve(self) -> list[float]: ...
    @property
    def time_total(self) -> float: ...

def solve(
    categories: NDArray[np.uint32],
    y: NDArray[np.float64],
    config: CG | GMRES | None = None,
    weights: NDArray[np.float64] | None = None,
    preconditioner: AdditiveSchwarz
    | MultiplicativeSchwarz
    | Preconditioner
    | None = None,
) -> SolveResult:
    """Solve fixed-effects normal equations.

    ``categories`` should be F-contiguous (column-major) for best performance.
    Use ``np.asfortranarray(categories)`` to convert.  A ``UserWarning`` is
    emitted when a C-contiguous array is passed.

    ``preconditioner`` controls the preconditioner:
    - ``None`` (default) — additive Schwarz with default local solver
    - ``Preconditioner.Off`` — unpreconditioned
    - ``AdditiveSchwarz(...)`` / ``MultiplicativeSchwarz(...)`` — advanced config
    """
    ...

def solve_batch(
    categories: NDArray[np.uint32],
    Y: NDArray[np.float64],
    config: CG | GMRES | None = None,
    weights: NDArray[np.float64] | None = None,
    preconditioner: AdditiveSchwarz
    | MultiplicativeSchwarz
    | Preconditioner
    | None = None,
) -> BatchSolveResult:
    """Solve fixed-effects normal equations for multiple RHS vectors.

    ``Y`` is a 2-D array of shape ``(n_obs, k)`` where each column is a
    separate response vector.  All solves share the same preconditioner
    and run in parallel via rayon.

    Same preconditioner options as :func:`solve`.
    """
    ...

class FePreconditioner:
    """A pre-built preconditioner (picklable).

    Obtained via ``Solver.preconditioner()``.  Pass it back to a new
    ``Solver(..., preconditioner=p)`` to skip the expensive factorisation.
    """

    def apply(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the preconditioner: ``y = M⁻¹ x``."""
        ...
    @property
    def nrows(self) -> int: ...
    @property
    def ncols(self) -> int: ...
    @staticmethod
    def from_bytes(data: bytes) -> FePreconditioner: ...

class Solver:
    """Persistent solver that reuses preconditioners across multiple solves.

    Build once, then call ``solve()`` or ``solve_batch()`` repeatedly.
    The expensive preconditioner factorization happens only at construction.
    """

    def __init__(
        self,
        categories: NDArray[np.uint32],
        config: CG | GMRES | None = None,
        weights: NDArray[np.float64] | None = None,
        preconditioner: AdditiveSchwarz
        | MultiplicativeSchwarz
        | Preconditioner
        | FePreconditioner
        | None = None,
    ) -> None: ...
    def solve(self, y: NDArray[np.float64]) -> SolveResult: ...
    def solve_batch(self, Y: NDArray[np.float64]) -> BatchSolveResult: ...
    def preconditioner(self) -> FePreconditioner | None:
        """Return the built preconditioner, or None if unconfigured."""
        ...
    @property
    def n_dofs(self) -> int: ...
    @property
    def n_obs(self) -> int: ...

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
