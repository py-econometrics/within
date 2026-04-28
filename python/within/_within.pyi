"""Type stubs for the ``within._within`` Rust extension module.

This module is the compiled PyO3 bridge to the ``within`` Rust crate.
Most users should import from ``within`` directly rather than from
``within._within``.
"""

import numpy as np
from numpy.typing import NDArray
from enum import IntEnum

class OperatorRepr(IntEnum):
    """How the Gramian operator (D^T W D) is represented internally.

    Attributes:
        Implicit: Matrix-free operator; lower memory, preferred for large problems.
        Explicit: Pre-assembled sparse CSR matrix; faster per iteration but uses more memory.
    """

    Implicit = 0
    Explicit = 1

class Preconditioner(IntEnum):
    """Preconditioner selection for CG / GMRES solvers.

    Use the enum variants for defaults, or pass an ``AdditiveSchwarz`` /
    ``MultiplicativeSchwarz`` instance for fine-grained control.

    Attributes:
        Additive: Additive Schwarz (default). Symmetric, works with CG and GMRES.
        Multiplicative: Multiplicative Schwarz. Non-symmetric, GMRES only.
        Off: No preconditioner. Useful for debugging or well-conditioned problems.
    """

    Additive = 0
    Multiplicative = 1
    Off = 2

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

class AdditiveSchwarzDiagnostics:
    """Diagnostics for an additive Schwarz preconditioner.

    Provides insight into the parallel structure and work distribution
    across subdomains. Useful for performance tuning.
    """

    @property
    def reduction_strategy(self) -> ReductionStrategy:
        """The configured reduction strategy (may be ``Auto``)."""
        ...
    @property
    def resolved_reduction_strategy(self) -> ReductionStrategy:
        """The actual strategy chosen after resolving ``Auto``."""
        ...
    @property
    def total_inner_parallel_work(self) -> int:
        """Sum of inner parallel work across all subdomains."""
        ...
    @property
    def max_inner_parallel_work(self) -> int:
        """Maximum inner parallel work in any single subdomain."""
        ...
    @property
    def total_scatter_dofs(self) -> int:
        """Total number of DOFs scattered across subdomains (with overlap)."""
        ...
    @property
    def outer_parallel_capacity(self) -> float:
        """Ratio indicating how well the outer loop parallelises (higher is better)."""
        ...
    @property
    def scatter_overlap(self) -> float:
        """Ratio of scattered DOFs to unique DOFs (1.0 = no overlap)."""
        ...
    def __repr__(self) -> str: ...

class CG:
    """Conjugate Gradient solver configuration.

    CG requires a symmetric preconditioner, so it cannot be used with
    ``Preconditioner.Multiplicative``. Use ``GMRES`` instead for
    non-symmetric preconditioners.

    Attributes:
        tol: Convergence tolerance on the relative residual norm. Default ``1e-8``.
        maxiter: Maximum number of Krylov iterations. Default ``1000``.
        operator: How the Gramian operator is represented. Default ``Implicit``.
        max_refinements: Maximum iterative-refinement steps after convergence.
            Default ``2``.
    """

    tol: float
    maxiter: int
    operator: OperatorRepr
    max_refinements: int
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        operator: OperatorRepr = OperatorRepr.Implicit,
        max_refinements: int = 2,
    ) -> None: ...

class GMRES:
    """GMRES (Generalized Minimal Residual) solver configuration.

    GMRES supports both symmetric and non-symmetric preconditioners,
    so it works with all ``Preconditioner`` variants.

    Attributes:
        tol: Convergence tolerance on the relative residual norm. Default ``1e-8``.
        maxiter: Maximum number of Krylov iterations. Default ``1000``.
        restart: Restart the Arnoldi process every this many iterations.
            Default ``30``.
        operator: How the Gramian operator is represented. Default ``Implicit``.
        max_refinements: Maximum iterative-refinement steps after convergence.
            Default ``2``.
    """

    tol: float
    maxiter: int
    restart: int
    operator: OperatorRepr
    max_refinements: int
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        restart: int = 30,
        operator: OperatorRepr = OperatorRepr.Implicit,
        max_refinements: int = 2,
    ) -> None: ...

class LSMR:
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
        converged: Whether the Krylov solver met the convergence tolerance.
        iterations: Total number of Krylov iterations performed, including
            iterative-refinement correction solves.
            iterative-refinement correction solves.
        residual: Final relative residual norm.
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
        iterations: Total Krylov iterations for each RHS.
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
    config: CG | GMRES | LSMR | None = None,
    weights: NDArray[np.float64] | None = None,
    preconditioner: AdditiveSchwarz
    | MultiplicativeSchwarz
    | Preconditioner
    | None = None,
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
        config: Krylov solver configuration. Pass ``CG()``, ``GMRES()``,
            or ``LSMR()``
            to override defaults (tolerance, max iterations, etc.).
            Default: ``CG(tol=1e-8, maxiter=1000)``.
        weights: Observation weights, shape ``(n_obs,)``, dtype ``float64``.
            Default: unit weights (unweighted).
        preconditioner: Controls preconditioning. ``None`` (default) uses
            additive Schwarz. ``Preconditioner.Off`` disables it.
            Pass ``AdditiveSchwarz(...)`` or ``MultiplicativeSchwarz(...)``
            for fine-grained control over local solvers.

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
    config: CG | GMRES | LSMR | None = None,
    weights: NDArray[np.float64] | None = None,
    preconditioner: AdditiveSchwarz
    | MultiplicativeSchwarz
    | Preconditioner
    | None = None,
) -> BatchSolveResult:
    """Solve fixed-effects normal equations for multiple response vectors.

    Builds the operator and preconditioner once, then solves for each column
    of ``Y`` in parallel via rayon. More efficient than calling ``solve()``
    in a loop because setup cost is shared.

    Args:
        categories: Factor assignments, shape ``(n_obs, n_factors)``,
            dtype ``uint32``. Same format as :func:`solve`.
        Y: Response matrix, shape ``(n_obs, k)``, dtype ``float64``.
            Each column is a separate response vector.
        config: Krylov solver configuration. Default: ``CG(tol=1e-8, maxiter=1000)``.
        weights: Observation weights, shape ``(n_obs,)``. Default: unit weights.
        preconditioner: Same options as :func:`solve`.

    Returns:
        A ``BatchSolveResult`` with per-RHS coefficients, convergence info,
        and timing.

    Example::

        import numpy as np
        import within

        cats = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint32)
        Y = np.column_stack([np.array([1.0, 2.0, 3.0, 4.0]),
                             np.array([4.0, 3.0, 2.0, 1.0])])
        result = within.solve_batch(cats, Y)
        print(result.x.shape)  # (n_dofs, 2)
    """
    ...

class FePreconditioner:
    """A pre-built Schwarz preconditioner that can be pickled and reused.

    Obtained via ``Solver.preconditioner()``. Pass it back to a new
    ``Solver(..., preconditioner=p)`` to skip the expensive factorisation
    step, which is especially useful when solving for different response
    vectors on the same panel structure.

    The object supports Python's ``pickle`` protocol for serialisation.
    """

    def apply(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the preconditioner: ``y = M^{-1} x``.

        Args:
            x: Input vector, shape ``(n_dofs,)``.

        Returns:
            Output vector, shape ``(n_dofs,)``.
        """
        ...
    @property
    def nrows(self) -> int:
        """Number of rows (DOFs) in the preconditioner."""
        ...
    @property
    def ncols(self) -> int:
        """Number of columns (DOFs) in the preconditioner."""
        ...
    @property
    def n_subdomains(self) -> int:
        """Number of Schwarz subdomains."""
        ...
    @property
    def subdomain_inner_parallel_work(self) -> list[int]:
        """Estimated inner parallel work per subdomain."""
        ...
    def additive_schwarz_diagnostics(self) -> AdditiveSchwarzDiagnostics | None:
        """Return diagnostics if this is an additive preconditioner, else ``None``."""
        ...
    def __init__(self, data: bytes) -> None:
        """Construct from serialised bytes (used by pickle)."""
        ...
    def __repr__(self) -> str: ...

class Solver:
    """Persistent solver that reuses preconditioners across multiple solves.

    Construct once with panel structure (categories, weights, config), then
    call :meth:`solve` or :meth:`solve_batch` repeatedly with different
    response vectors. The expensive preconditioner factorisation happens
    only at construction time.

    Lifecycle::

        solver = within.Solver(categories, weights=w)
        r1 = solver.solve(y1)
        r2 = solver.solve(y2)

        # Save and reuse the preconditioner for a new Solver:
        p = solver.preconditioner()
        solver2 = within.Solver(categories, weights=w, preconditioner=p)
    """

    def __init__(
        self,
        categories: NDArray[np.uint32],
        config: CG | GMRES | LSMR | None = None,
        weights: NDArray[np.float64] | None = None,
        preconditioner: AdditiveSchwarz
        | MultiplicativeSchwarz
        | Preconditioner
        | FePreconditioner
        | None = None,
    ) -> None:
        """Build a solver for the given panel structure.

        Args:
            categories: Factor assignments, shape ``(n_obs, n_factors)``,
                dtype ``uint32``. Should be F-contiguous for best performance.
            config: Krylov solver configuration. Default: ``CG(tol=1e-8, maxiter=1000)``.
            weights: Observation weights, shape ``(n_obs,)``. Default: unit weights.
            preconditioner: Preconditioner configuration. Pass a
                ``FePreconditioner`` from a previous ``Solver`` to reuse it.
                Default: additive Schwarz.
        """
        ...
    def solve(self, y: NDArray[np.float64]) -> SolveResult:
        """Solve for a single response vector.

        Args:
            y: Response vector, shape ``(n_obs,)``, dtype ``float64``.

        Returns:
            A ``SolveResult`` with coefficients and convergence info.
        """
        ...
    def solve_batch(self, Y: NDArray[np.float64]) -> BatchSolveResult:
        """Solve for multiple response vectors in parallel.

        Args:
            Y: Response matrix, shape ``(n_obs, k)``, dtype ``float64``.

        Returns:
            A ``BatchSolveResult`` with per-RHS results.
        """
        ...
    def preconditioner(self) -> FePreconditioner | None:
        """Return the built preconditioner, or ``None`` if unconfigured.

        The returned object is picklable and can be passed to a new
        ``Solver(..., preconditioner=p)`` to skip the expensive build step.
        """
        ...
    @property
    def n_dofs(self) -> int:
        """Number of DOFs (coefficients) in the model."""
        ...
    @property
    def n_obs(self) -> int:
        """Number of observations."""
        ...

# ---------------------------------------------------------------------------
# Advanced config classes (for benchmarks / power users)
# ---------------------------------------------------------------------------

class ApproxCholConfig:
    """Configuration for approximate Cholesky factorisation in local solvers.

    Attributes:
        seed: Random seed for the approximate factorisation. Default ``0``.
        split: Number of split-merge passes. ``1`` means no splitting.
            Must be >= 1. Default ``1``.
    """

    seed: int
    split: int
    def __init__(self, seed: int = 0, split: int = 1) -> None: ...

class ApproxSchurConfig:
    """Configuration for approximate Schur complement factorisation.

    Attributes:
        seed: Random seed for the approximate factorisation. Default ``0``.
        split: Number of split passes. Must be >= 1. Default ``1``.
    """

    seed: int
    split: int
    def __init__(self, seed: int = 0, split: int = 1) -> None: ...

class SchurComplement:
    """Schur-complement local solver for Schwarz subdomains.

    Reduces each subdomain system via a Schur complement, then solves
    the reduced system. Efficient when subdomains have a natural
    interior/boundary partition.

    Attributes:
        approx_chol: Approximate Cholesky config for the interior block.
            Default: exact factorisation (``None``).
        approx_schur: Approximate Schur complement config. Default: exact (``None``).
        dense_threshold: Subdomains with fewer DOFs than this threshold use
            a dense direct solve instead of sparse factorisation. Default ``24``.
    """

    approx_chol: ApproxCholConfig | None
    approx_schur: ApproxSchurConfig | None
    dense_threshold: int
    def __init__(
        self,
        approx_chol: ApproxCholConfig | None = None,
        approx_schur: ApproxSchurConfig | None = None,
        dense_threshold: int = 24,
    ) -> None: ...

class AdditiveSchwarz:
    """Additive Schwarz preconditioner with configurable local solver.

    Symmetric, so it works with both CG and GMRES. This is the recommended
    preconditioner for most problems.

    Attributes:
        local_solver: Local solver for each subdomain. Pass ``SchurComplement()``
            to customise, or ``None`` for the default.
        reduction: Strategy for combining subdomain contributions.
            Default ``ReductionStrategy.Auto``.
    """

    local_solver: SchurComplement | None
    reduction: ReductionStrategy
    def __init__(
        self,
        local_solver: SchurComplement | None = None,
        reduction: ReductionStrategy = ReductionStrategy.Auto,
    ) -> None: ...

class MultiplicativeSchwarz:
    """Multiplicative Schwarz preconditioner with configurable local solver.

    Non-symmetric, so it requires GMRES (not CG). May converge in fewer
    iterations than additive Schwarz but has less parallelism.

    Attributes:
        local_solver: Local solver for each subdomain. Pass ``SchurComplement()``
            to customise, or ``None`` for the default.
    """

    local_solver: SchurComplement | None
    def __init__(
        self,
        local_solver: SchurComplement | None = None,
    ) -> None: ...
