"""Type stubs for the ``within._within`` Rust extension module.

This module is the compiled PyO3 bridge to the ``within`` Rust crate.
Most users should import from ``within`` directly rather than from
``within._within``.
"""

import numpy as np
from numpy.typing import NDArray

class PreconditionerConfig:
    """Preconditioner selection shortcut for the LSMR solver.

    Not an ``Enum``: the members below are class attributes (not iterable, so
    ``list(PreconditionerConfig)`` raises). Use them for defaults, or pass an
    ``AdditiveSchwarz`` instance for fine-grained control.

    Attributes:
        Additive: Additive Schwarz (default).
        Off: No preconditioner. Useful for debugging or well-conditioned problems.
        Diagonal: Diagonal/Jacobi preconditioner.
    """

    Additive: PreconditionerConfig
    Off: PreconditionerConfig
    Diagonal: PreconditionerConfig

class ReductionStrategy:
    """Strategy for combining subdomain contributions in additive Schwarz.

    Not an ``Enum``: the members below are class attributes (not iterable).

    Attributes:
        Auto: Let the solver choose based on problem structure (recommended).
        AtomicScatter: Use atomic operations to scatter subdomain results.
        ParallelReduction: Use parallel reduction over subdomain results.
    """

    Auto: ReductionStrategy
    AtomicScatter: ReductionStrategy
    ParallelReduction: ReductionStrategy

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

    @property
    def tol(self) -> float: ...
    @property
    def maxiter(self) -> int: ...
    @property
    def local_size(self) -> int | None: ...
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        local_size: int | None = None,
    ) -> None: ...

class UnidentifiedDirection:
    """A per-level design direction the data cannot identify.

    Attributes:
        term: Index into the design's term list.
        level: Level index within the term (``0..n_levels``).
        column: Column within the term's per-level block — intercept first
            (when present), then slopes in declaration order.
    """

    @property
    def term(self) -> int: ...
    @property
    def level(self) -> int: ...
    @property
    def column(self) -> int: ...

class CoefficientLayout:
    """Translate a ``(term, level, column)`` coefficient address to its flat
    ``SolveResult.x`` index and back, so callers need not reconstruct the
    term-major offset formula.

    ``n_levels``, ``n_columns``, ``index``, and ``address`` raise ``IndexError``
    on an out-of-range coordinate rather than returning a wrong value.
    """

    def n_dofs(self) -> int: ...
    def n_terms(self) -> int: ...
    def n_levels(self, term: int) -> int: ...
    def n_columns(self, term: int) -> int: ...
    def index(self, term: int, level: int, column: int) -> int: ...
    def address(self, index: int) -> tuple[int, int, int]: ...

class SolveResult:
    """Result of a single fixed-effects solve.

    Attributes:
        x: Fixed-effect coefficients, shape ``(n_dofs,)``. Term-major:
            coefficient column ``c`` of level ``level`` sits at
            ``term_offset + c * n_levels + level``, columns ordered
            ``[intercept?, slopes...]`` (for plain factors: all levels of
            factor 0 first, then factor 1, etc.). Slots for unidentified
            directions hold the minimal-norm value ``0``, never NaN.
        unidentified: Per-level directions the data cannot identify, as
            :class:`UnidentifiedDirection` records.
        layout: Address <-> flat-``x``-index translation for the coefficients.
        demeaned: Response vector after subtracting estimated fixed effects,
            shape ``(n_obs,)``.
        converged: Whether the LSMR solver met the convergence tolerance.
        iterations: Total number of LSMR iterations performed.
        residual: Relative normal-equation residual
            ``||D^T W (y - Dx)|| / ||D^T W y||`` estimated from the LSMR
            recurrence at no extra cost. Exact for an unpreconditioned solve;
            measured in the preconditioner's metric otherwise.
        time_total: Wall-clock time for the entire solve (setup + solve), in seconds.
        time_setup: Wall-clock time for the setup phase (operator + preconditioner
            construction), in seconds.
        time_solve: Wall-clock time for the iterative solve phase, in seconds.
    """

    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def unidentified(self) -> list[UnidentifiedDirection]: ...
    @property
    def layout(self) -> CoefficientLayout: ...
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
            Slots for unidentified directions hold the minimal-norm value
            ``0``, never NaN.
        unidentified: Per-level directions the data cannot identify, as
            :class:`UnidentifiedDirection` records; shared across all RHS.
        layout: Address <-> flat-``x``-index translation for the coefficients.
        demeaned: Demeaned responses, shape ``(n_obs, k)`` (column-major).
        converged: Whether each RHS converged.
        iterations: Total LSMR iterations for each RHS.
        residual: Per-RHS relative normal-equation residual estimate
            (see ``SolveResult.residual``).
        time_solve: Wall-clock solve time for each RHS, in seconds.
        time_setup: Wall-clock time for the shared setup phase (solver and
            preconditioner construction), in seconds; 0 when a pre-built
            preconditioner was reused.
        time_total: Wall-clock time for the entire batch (including shared setup),
            in seconds.
    """

    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def unidentified(self) -> list[UnidentifiedDirection]: ...
    @property
    def layout(self) -> CoefficientLayout: ...
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
    def time_setup(self) -> float: ...
    @property
    def time_total(self) -> float: ...

class Effect:
    """One factor's effect: level codes, an optional intercept, and slope covariates."""

    def __init__(
        self,
        levels: NDArray[np.uint32],
        intercept: bool,
        slopes: list[NDArray[np.float64]] | None = None,
    ) -> None: ...

def solve(
    design: NDArray[np.uint32] | list[Effect],
    y: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    options: LsmrOptions | None = None,
    preconditioner: (
        PreconditionerConfig | AdditiveSchwarz | Preconditioner | None
    ) = None,
) -> SolveResult:
    """Solve fixed-effects normal equations for a single response vector.

    Computes the fixed-effect coefficients by solving the normal equations
    ``D^T W D x = D^T W y`` where ``D`` is the dummy-variable design matrix
    implied by ``categories`` and ``W`` is the diagonal weight matrix.

    Args:
        design: Either a ``(n_obs, n_factors)`` ``uint32`` array of factor
            assignments (F-contiguous for best performance; a ``UserWarning``
            is emitted otherwise), or a list of :class:`Effect` terms.
        y: Response vector, shape ``(n_obs,)``, dtype ``float64``.
        weights: Observation weights, shape ``(n_obs,)``, dtype ``float64``.
            Default: unit weights (unweighted).
        options: LSMR solver tuning. Pass ``LsmrOptions(...)`` to override
            defaults. Default: ``LsmrOptions(tol=1e-8, maxiter=1000)``.
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
        ValueError: If dimensions or values are inconsistent.
        TypeError: If an argument has the wrong type or dtype.
        RuntimeError: If the solve fails at runtime (poisoned lock, or a
            subdomain solve diverges).

    Note:
        A single-threaded run (``RAYON_NUM_THREADS=1``) is bitwise-reproducible.
        Across thread counts, parallel summation reorders floating-point adds,
        so coefficients differ at the ULP scale — reproducible within solver
        tolerance, not bitwise. Pin the thread count to keep estimates stable
        within solver tolerance across runs (only single-threaded is bitwise).

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
    design: NDArray[np.uint32] | list[Effect],
    Y: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    options: LsmrOptions | None = None,
    preconditioner: (
        PreconditionerConfig | AdditiveSchwarz | Preconditioner | None
    ) = None,
) -> BatchSolveResult:
    """Solve fixed-effects normal equations for multiple response vectors.

    Equivalent to calling :func:`solve` on each column of ``Y`` but amortises
    the setup phase (preconditioner construction).

    Args:
        design: Either a ``(n_obs, n_factors)`` ``uint32`` array of factor
            assignments (F-contiguous for best performance; a ``UserWarning``
            is emitted otherwise), or a list of :class:`Effect` terms.
        Y: Response matrix, shape ``(n_obs, k)``, dtype ``float64``. Each column
            is a separate response vector.
        weights: Observation weights. Default: unit weights.
        options: LSMR solver tuning. Default: ``LsmrOptions(tol=1e-8, maxiter=1000)``.
        preconditioner: Preconditioner configuration; see :func:`solve` for the
            accepted forms.

    Returns:
        A ``BatchSolveResult`` with stacked coefficients and per-RHS metadata.

    Raises:
        ValueError: If dimensions or values are inconsistent.
        TypeError: If an argument has the wrong type or dtype.
        RuntimeError: If the solve fails at runtime (poisoned lock, or a
            subdomain solve diverges).
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
        design: NDArray[np.uint32] | list[Effect],
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

    @property
    def seed(self) -> int: ...
    @property
    def split_merge(self) -> int | None: ...
    def __init__(self, seed: int = 0, split_merge: int | None = None) -> None: ...

class ApproxSchurConfig:
    """Configuration for approximate Schur complement via clique-tree sampling."""

    @property
    def seed(self) -> int: ...
    @property
    def split(self) -> int: ...
    def __init__(self, seed: int = 0, split: int = 1) -> None: ...

class Schur:
    """Schur-complement reduction mode for :class:`LocalSolverConfig`.

    Omitting ``schur`` on ``LocalSolverConfig`` uses the library default
    (approximate); use these static constructors to request a specific mode.
    """

    @staticmethod
    def approximate(config: ApproxSchurConfig | None = None) -> Schur: ...
    @staticmethod
    def exact() -> Schur: ...

class ScalingConfig:
    """Certification policy for the diagonal scaling of signed components.

    ``on_failure`` is ``"warn"`` (clamp residual deficits — preconditioner
    quality only — and emit a ``UserWarning``) or ``"error"`` (fail the build).
    """

    @property
    def tolerance(self) -> float: ...
    @property
    def max_sweeps(self) -> int: ...
    @property
    def on_failure(self) -> str: ...
    def __init__(
        self,
        tolerance: float | None = None,
        max_sweeps: int | None = None,
        on_failure: str | None = None,
    ) -> None: ...

class LocalSolverConfig:
    """Local solver: Schur reduction + approximate Cholesky.

    Omit ``schur`` for the library default (approximate Schur); pass
    ``Schur.exact()`` for the exact complement.
    """

    @property
    def approx_chol(self) -> ApproxCholConfig | None: ...
    @property
    def schur(self) -> Schur | None: ...
    @property
    def dense_threshold(self) -> int: ...
    @property
    def scaling(self) -> ScalingConfig | None: ...
    def __init__(
        self,
        approx_chol: ApproxCholConfig | None = None,
        schur: Schur | None = None,
        dense_threshold: int | None = None,
        scaling: ScalingConfig | None = None,
    ) -> None: ...

class AdditiveSchwarz:
    """Additive Schwarz preconditioner with configurable local solver."""

    @property
    def local_solver(self) -> LocalSolverConfig | None: ...
    @property
    def reduction(self) -> ReductionStrategy: ...
    def __init__(
        self,
        local_solver: LocalSolverConfig | None = None,
        reduction: ReductionStrategy = ReductionStrategy.Auto,
    ) -> None: ...
