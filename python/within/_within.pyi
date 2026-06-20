"""Type stubs for the ``within._within`` Rust extension module.

This module is the compiled PyO3 bridge to the ``within`` Rust crate.
Most users should import from ``within`` directly rather than from
``within._within``.
"""

import numpy as np
from numpy.typing import NDArray
from enum import IntEnum

class PreconditionerConfig(IntEnum):
    """Preconditioner selection shortcut for the LSMR solver.

    Use the enum variants for defaults, or pass an ``AdditiveSchwarz``
    instance for fine-grained control.

    Attributes:
        Additive: Additive Schwarz (default).
        Off: No preconditioner. Useful for debugging or well-conditioned problems.
        Diagonal: Diagonal/Jacobi preconditioner.
    """

    Additive = 0
    Off = 1
    Diagonal = 2

class ReductionStrategy(IntEnum):
    """Strategy for combining subdomain contributions in additive Schwarz.

    Attributes:
        Auto: Let the solver choose based on problem structure (recommended).
        AtomicScatter: Use atomic operations to scatter subdomain results.
        ParallelReduction: Use parallel reduction over subdomain results.
    """

    Auto = 0
    AtomicScatter = 1
    ParallelReduction = 2

class LsmrOptions:
    """Modified LSMR solver configuration.

    Uses Modified Golub-Kahan bidiagonalization to solve the rectangular
    least-squares problem directly. The preconditioner ``M ≈ A^T A`` is
    applied as a single ``M^{-1}`` solve per iteration — no square-root
    factorization needed.

    Attributes:
        tol: Convergence tolerance. Default ``1e-8``.
        maxiter: Maximum number of iterations. Default ``1000``.
        local_size: Number of past ``v`` vectors to reorthogonalize against
            via windowed modified Gram-Schmidt. ``None`` (default) disables —
            the plain short recurrence is used. ``5..20`` is cheap insurance
            for ill-conditioned problems where rounding causes the
            bidiagonalization to lose orthogonality and convergence to
            stall. Memory cost is ``local_size * n_dofs`` doubles
            unpreconditioned, ``2 * local_size * n_dofs`` preconditioned.
    """

    tol: float
    maxiter: int
    local_size: int | None
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        local_size: int | None = None,
    ) -> None: ...

class SolveResult:
    """Result of a single fixed-effects solve.

    Attributes:
        x: Fixed-effect coefficients, shape ``(n_dofs,)``. The DOF ordering
            matches the factor levels: all levels of factor 0 first, then
            factor 1, etc.
        demeaned: Response vector after subtracting estimated fixed effects,
            shape ``(n_obs,)``.
        converged: Whether the LSMR solver met the convergence tolerance.
        iterations: Total number of LSMR iterations performed.
        residual: Final relative residual norm.
        time_total: Wall-clock time for the entire solve (setup + solve), in seconds.
        time_setup: Wall-clock time for the setup phase (operator + preconditioner
            construction), in seconds.
        time_solve: Wall-clock time for the iterative solve phase, in seconds.
    """

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
    """Result of a batch solve across multiple response vectors.

    Per-RHS fields are lists of length ``k`` (one entry per column of ``Y``).

    Attributes:
        x: Fixed-effect coefficients, shape ``(n_dofs, k)`` (column-major).
        demeaned: Demeaned responses, shape ``(n_obs, k)`` (column-major).
        converged: Whether each RHS converged.
        iterations: Total LSMR iterations for each RHS.
        residual: Final relative residual norm for each RHS.
        time_solve: Wall-clock solve time for each RHS, in seconds.
        time_total: Wall-clock time for the entire batch (including shared setup),
            in seconds.
    """

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
    options: LsmrOptions | None = None,
    weights: NDArray[np.float64] | None = None,
    preconditioner: (
        PreconditionerConfig | AdditiveSchwarz | Preconditioner | None
    ) = None,
) -> SolveResult:
    """Solve fixed-effects normal equations for a single response vector.

    Computes the fixed-effect coefficients by solving the normal equations
    ``D^T W D x = D^T W y`` where ``D`` is the dummy-variable design matrix
    implied by ``categories`` and ``W`` is the diagonal weight matrix.

    Args:
        categories: Factor assignments, shape ``(n_obs, n_factors)``,
            dtype ``uint32``. Each column contains zero-based level indices
            for one factor. Should be F-contiguous (column-major) for best
            performance; a ``UserWarning`` is emitted for C-contiguous arrays.
        y: Response vector, shape ``(n_obs,)``, dtype ``float64``.
        options: LSMR solver tuning. Pass ``LsmrOptions(...)`` to override
            defaults. Default: ``LsmrOptions(tol=1e-8, maxiter=1000)``.
        weights: Observation weights, shape ``(n_obs,)``, dtype ``float64``.
            Default: unit weights (unweighted).
        preconditioner: Controls preconditioning. Five input forms are accepted:
            ``None`` (default) builds the additive Schwarz preconditioner with
            default settings. ``PreconditionerConfig.Off`` disables it.
            ``PreconditionerConfig.Diagonal`` uses diagonal/Jacobi scaling.
            ``AdditiveSchwarz(...)`` overrides the local-solver / reduction
            settings. A previously-built ``Preconditioner`` instance reuses an
            existing factorisation.

    Returns:
        A ``SolveResult`` with coefficients, demeaned response, convergence
        info, and timing breakdown.

    Raises:
        ValueError: If dimensions are inconsistent or the solve fails.

    Example::

        import numpy as np
        import within

        categories = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = within.solve(categories, y)
        print(result.x)         # fixed-effect coefficients
        print(result.converged) # True
    """
    ...

def solve_batch(
    categories: NDArray[np.uint32],
    Y: NDArray[np.float64],
    options: LsmrOptions | None = None,
    weights: NDArray[np.float64] | None = None,
    preconditioner: (
        PreconditionerConfig | AdditiveSchwarz | Preconditioner | None
    ) = None,
) -> BatchSolveResult:
    """Solve fixed-effects normal equations for multiple response vectors.

    Equivalent to calling :func:`solve` on each column of ``Y`` but amortises
    the setup phase (preconditioner construction).

    Args:
        categories: Factor assignments, shape ``(n_obs, n_factors)``, dtype ``uint32``.
        Y: Response matrix, shape ``(n_obs, k)``, dtype ``float64``. Each column
            is a separate response vector.
        options: LSMR solver tuning. Default: ``LsmrOptions(tol=1e-8, maxiter=1000)``.
        weights: Observation weights. Default: unit weights.
        preconditioner: Preconditioner configuration; see :func:`solve` for the
            accepted forms.

    Returns:
        A ``BatchSolveResult`` with stacked coefficients and per-RHS metadata.

    Raises:
        ValueError: If dimensions are inconsistent.
    """
    ...

class Preconditioner:
    """Pre-built fixed-effects preconditioner.

    Built once per design and reused across solves via the persistent
    :class:`Solver`. Pickleable for offline construction; can also be
    deserialised manually via ``Preconditioner(bytes_payload)`` (the same
    payload produced by ``__reduce__`` / ``pickle.dumps``).
    """

    def __init__(self, data: bytes) -> None: ...
    def apply(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the preconditioner: ``y = M⁻¹ x``."""
        ...
    @property
    def nrows(self) -> int: ...
    @property
    def ncols(self) -> int: ...
    def __repr__(self) -> str: ...
    def __reduce__(self) -> tuple: ...

class Solver:
    """Persistent solver with cached preconditioner.

    Build once with the design matrix; call :meth:`solve` or
    :meth:`solve_batch` repeatedly with different response vectors.
    """

    def __init__(
        self,
        categories: NDArray[np.uint32],
        weights: NDArray[np.float64] | None = None,
        preconditioner: (
            PreconditionerConfig | AdditiveSchwarz | Preconditioner | None
        ) = None,
    ) -> None: ...
    def solve(
        self,
        y: NDArray[np.float64],
        options: LsmrOptions | None = None,
    ) -> SolveResult:
        """Solve for a single response vector with the given LSMR tuning."""
        ...
    def solve_batch(
        self,
        Y: NDArray[np.float64],
        options: LsmrOptions | None = None,
    ) -> BatchSolveResult:
        """Solve for multiple response vectors in parallel."""
        ...
    @property
    def preconditioner(self) -> Preconditioner | None:
        """Access the cached preconditioner (for serialization or reuse)."""
        ...
    @property
    def n_dofs(self) -> int: ...
    @property
    def n_obs(self) -> int: ...

# ---------------------------------------------------------------------------
# Local solver configuration (advanced)
# ---------------------------------------------------------------------------

class ApproxCholConfig:
    """Configuration for approximate Cholesky factorization."""

    seed: int
    split_merge: int | None
    def __init__(self, seed: int = 0, split_merge: int | None = None) -> None: ...

class ApproxSchurConfig:
    """Configuration for approximate Schur complement via clique-tree sampling."""

    seed: int
    split: int
    def __init__(self, seed: int = 0, split: int = 1) -> None: ...

class LocalSolverConfig:
    """Local solver: Schur reduction + approximate Cholesky.

    Note: the Python-level constructor exposed via ``within.config`` uses an
    explicit default for ``approx_schur`` (the library-default approximate
    config). The native PyO3 constructor accepts ``None`` as the *explicit*
    "exact Schur" signal — see ``python/within/config.py`` for the wrapper
    that injects the default.
    """

    approx_chol: ApproxCholConfig | None
    approx_schur: ApproxSchurConfig | None
    dense_threshold: int
    def __init__(
        self,
        approx_chol: ApproxCholConfig | None = None,
        approx_schur: ApproxSchurConfig | None = None,
        dense_threshold: int | None = None,
    ) -> None: ...

class AdditiveSchwarz:
    """Additive Schwarz preconditioner with configurable local solver."""

    local_solver: LocalSolverConfig | None
    reduction: ReductionStrategy
    def __init__(
        self,
        local_solver: LocalSolverConfig | None = None,
        reduction: ReductionStrategy = ReductionStrategy.Auto,
    ) -> None: ...
