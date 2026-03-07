"""Core data types and helpers for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from within import CG, GMRES, solve
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    MultiplicativeSchwarz,
    SchurComplement,
)


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
    converged: bool = False
    passed: bool | None = None  # For correctness suites
    phase_timings: dict[str, float] = field(default_factory=dict)  # For profiling
    extra: dict[str, Any] = field(default_factory=dict)  # Suite-specific data


def run_solve(
    categories: list[NDArray[np.int64]],
    n_levels: list[int],
    y: NDArray[np.float64],
    config: SolverConfig,
) -> BenchmarkResult:
    """Solve using *config* and return a timed result."""
    result = solve(
        np.column_stack(categories).astype(np.uintp),
        y,
        config.config,
        n_levels=n_levels,
    )

    return BenchmarkResult(
        problem="",  # Caller fills this in
        config=config.label,
        n_dofs=sum(n_levels),
        n_rows=len(categories[0]),
        setup_time=result.time_setup,
        solve_time=result.time_solve,
        iterations=result.iterations,
        final_residual=result.residual,
        converged=result.converged,
    )


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
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=schur),
            ),
        ),
        SolverConfig(
            "GMRES(Mult-Schwarz)",
            GMRES(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=MultiplicativeSchwarz(local_solver=schur),
            ),
        ),
    ]


def run_problem_set(
    problems: list[ProblemSpec],
    configs: list[SolverConfig],
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
                result = run_solve(cats, n_levels, y, cfg)
                result.problem = prob.name
                all_results.append(result)
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f"  WARNING: {cfg.label} failed: {e}")
    return all_results
