"""Benchmark framework: types, solve helpers, and suite registry.

Public API
----------
- ``BenchmarkResult`` — result dataclass
- ``ProblemSpec`` — problem specification
- ``SolverConfig`` — solver configuration wrapper
- ``SuiteOptions`` — options for suite runs
- ``SuiteInfo`` — metadata + callable for one benchmark suite
- ``suite`` — decorator to register a suite
- ``list_suites`` / ``get_suite`` — suite lookup
- ``run_solve`` / ``run_problem_set`` — solve helpers
- ``standard_solver_configs`` — default CG + GMRES configs
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from within import CG, GMRES, solve
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    MultiplicativeSchwarz,
    ReductionStrategy,
    SchurComplement,
)

ScaleProfile = Literal["smoke", "iterate", "full"]
_T = TypeVar("_T")
BENCHMARK_SOLVER_TOL_MIN = 1e-7

# ---------------------------------------------------------------------------
# Core data types (formerly _types.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProblemSpec:
    """Specification for a benchmark problem."""

    name: str
    generator: str  # Registry key in _problems.py
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for a solve via the Rust-backed API."""

    label: str
    config: CG | GMRES
    preconditioner: Any = None


@dataclass
class BenchmarkResult:
    """Result from a single (problem, solver config) benchmark run."""

    problem: str
    config: str  # SolverConfig.label
    n_dofs: int
    n_rows: int
    setup_time: float = 0.0
    solve_time: float = 0.0
    iterations: int = 0
    final_residual: float = 0.0
    demeaning_error: float = 0.0
    converged: bool = False
    passed: bool | None = None  # For correctness suites


def max_abs_group_mean(
    categories: list[NDArray[np.int64]],
    n_levels: list[int],
    demeaned: NDArray[np.float64],
) -> float:
    """Max absolute group mean of demeaned values across all factor-levels.

    If demeaning is perfect, the mean of demeaned values within every level
    of every factor is zero.  This returns the worst-case violation.
    """
    worst = 0.0
    for cats, nl in zip(categories, n_levels):
        sums = np.zeros(nl)
        counts = np.zeros(nl)
        np.add.at(sums, cats, demeaned)
        np.add.at(counts, cats, 1.0)
        means = sums / np.maximum(counts, 1.0)
        worst = max(worst, float(np.abs(means).max()))
    return worst


def _solve_once(
    categories: list[NDArray[np.int64]],
    n_levels: list[int],
    y: NDArray[np.float64],
    config: SolverConfig,
) -> BenchmarkResult:
    """Run one timed solve."""
    result = solve(
        np.asfortranarray(np.column_stack(categories).astype(np.uint32)),
        y,
        config.config,
        preconditioner=config.preconditioner,
    )

    demean_err = max_abs_group_mean(categories, n_levels, result.demeaned)

    return BenchmarkResult(
        problem="",  # Caller fills this in
        config=config.label,
        n_dofs=sum(n_levels),
        n_rows=len(categories[0]),
        setup_time=result.time_setup,
        solve_time=result.time_solve,
        iterations=result.iterations,
        final_residual=result.residual,
        demeaning_error=demean_err,
        converged=result.converged,
    )


def _aggregate_runs(runs: list[BenchmarkResult]) -> BenchmarkResult:
    """Aggregate repeated solves into one median benchmark result."""
    if not runs:
        raise ValueError("expected at least one benchmark run")

    first = runs[0]
    return BenchmarkResult(
        problem=first.problem,
        config=first.config,
        n_dofs=first.n_dofs,
        n_rows=first.n_rows,
        setup_time=float(statistics.median(r.setup_time for r in runs)),
        solve_time=float(statistics.median(r.solve_time for r in runs)),
        iterations=int(round(statistics.median(r.iterations for r in runs))),
        final_residual=float(statistics.median(r.final_residual for r in runs)),
        demeaning_error=max(r.demeaning_error for r in runs),
        converged=all(r.converged for r in runs),
    )


def run_solve(
    categories: list[NDArray[np.int64]],
    n_levels: list[int],
    y: NDArray[np.float64],
    config: SolverConfig,
    opts: SuiteOptions | None = None,
) -> BenchmarkResult:
    """Solve using *config* and return a timed result.

    When *opts* requests repeats, discard warmups and report medians.
    """
    if opts is None:
        return _solve_once(categories, n_levels, y, config)

    runs: list[BenchmarkResult] = []
    total_runs = opts.warmup + opts.repeat
    for run_idx in range(total_runs):
        result = _solve_once(categories, n_levels, y, config)
        if run_idx >= opts.warmup:
            runs.append(result)
    return _aggregate_runs(runs)


def benchmark_solver_tol(tol: float) -> float:
    """Benchmark tolerance floor used to avoid borderline non-convergence noise."""
    return max(tol, BENCHMARK_SOLVER_TOL_MIN)


def benchmark_cg(opts: Any, *, maxiter: int | None = None) -> CG:
    """Construct a CG config with benchmark-standard tolerance handling."""
    return CG(
        tol=benchmark_solver_tol(opts.tol),
        maxiter=opts.maxiter if maxiter is None else maxiter,
    )


def benchmark_gmres(opts: Any, *, maxiter: int | None = None) -> GMRES:
    """Construct a GMRES config with benchmark-standard tolerance handling."""
    return GMRES(
        tol=benchmark_solver_tol(opts.tol),
        maxiter=opts.maxiter if maxiter is None else maxiter,
    )


def make_additive_schwarz(local_solver: Any) -> AdditiveSchwarz:
    """Construct additive Schwarz using the default Auto reduction mode."""
    return AdditiveSchwarz(local_solver=local_solver, reduction=ReductionStrategy.Auto)


def standard_solver_configs(opts: Any) -> list[SolverConfig]:
    """Standard CG + GMRES solver configs used by most benchmark suites.

    *opts* must have ``seed``, ``tol``, and ``maxiter`` attributes
    (typically a ``SuiteOptions``).
    """
    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    return [
        SolverConfig(
            "CG(Schwarz)",
            benchmark_cg(opts),
            preconditioner=make_additive_schwarz(local_solver=schur),
        ),
        SolverConfig(
            "GMRES(Mult-Schwarz)",
            benchmark_gmres(opts),
            preconditioner=MultiplicativeSchwarz(local_solver=schur),
        ),
    ]


def run_problem_set(
    problems: list[ProblemSpec],
    configs: list[SolverConfig],
    opts: SuiteOptions | None = None,
) -> list[BenchmarkResult]:
    """Run a list of ``ProblemSpec`` through the given solver configs.

    Prints progress and catches non-fatal exceptions per solver.
    """
    from ._problems import get_generator

    all_results: list[BenchmarkResult] = []
    for prob in problems:
        gen = get_generator(prob.generator)
        cats, n_levels, y = gen(**prob.params, seed=prob.seed)
        n_fe = len(n_levels)
        print(
            f"\nProblem: {prob.name}  ({n_fe}-FE, DOFs={sum(n_levels)}, Rows={len(cats[0])})"
        )

        for cfg in configs:
            try:
                result = run_solve(cats, n_levels, y, cfg, opts)
                result.problem = prob.name
                all_results.append(result)
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f"  WARNING: {cfg.label} failed: {e}")
    return all_results


# ---------------------------------------------------------------------------
# Suite registry (formerly _registry.py)
# ---------------------------------------------------------------------------


@dataclass
class SuiteOptions:
    """Options passed to every suite run function."""

    seed: int = 42
    tol: float = 1e-8
    maxiter: int = 2000
    profile: ScaleProfile = "full"
    repeat: int = 1
    warmup: int = 0

    @property
    def quick(self) -> bool:
        return self.profile == "smoke"

    def select(
        self,
        *,
        smoke: list[_T],
        iterate: list[_T] | None = None,
        full: list[_T] | None = None,
    ) -> list[_T]:
        """Pick benchmark cases for the active profile.

        `smoke` is the smallest, `iterate` is the default development tier,
        and `full` is the long validation tier.
        """
        if self.profile == "smoke":
            return smoke
        if self.profile == "iterate":
            if iterate is not None:
                return iterate
            if full is not None:
                return full
            return smoke
        if full is not None:
            return full
        if iterate is not None:
            return iterate
        return smoke


@dataclass
class SuiteInfo:
    """Metadata + callable for one benchmark suite."""

    name: str
    description: str
    tags: frozenset[str]
    run_fn: Callable[[SuiteOptions], list[BenchmarkResult]]


_SUITES: dict[str, SuiteInfo] = {}


def suite(
    name: str,
    description: str = "",
    tags: tuple[str, ...] | frozenset[str] = (),
) -> Callable[
    [Callable[[SuiteOptions], list[BenchmarkResult]]],
    Callable[[SuiteOptions], list[BenchmarkResult]],
]:
    """Decorator that registers a function as a benchmark suite."""
    tag_set = frozenset(tags)

    def _decorator(
        fn: Callable[[SuiteOptions], list[BenchmarkResult]],
    ) -> Callable[[SuiteOptions], list[BenchmarkResult]]:
        if name in _SUITES:
            raise ValueError(f"Duplicate suite name: {name!r}")
        _SUITES[name] = SuiteInfo(
            name=name,
            description=description,
            tags=tag_set,
            run_fn=fn,
        )
        return fn

    return _decorator


def list_suites() -> dict[str, SuiteInfo]:
    """Return all registered suites."""
    return dict(_SUITES)


def get_suite(name: str) -> SuiteInfo:
    """Look up a suite by name."""
    if name not in _SUITES:
        raise KeyError(f"Unknown suite {name!r}. Available: {sorted(_SUITES)}")
    return _SUITES[name]
