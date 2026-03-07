"""Iteration reduction suite."""

from __future__ import annotations

from within import CG, GMRES
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    MultiplicativeSchwarz,
    SchurComplement,
)
from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solvers import run_solve
from .._table import print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig


@suite(
    "iteration_reduction",
    description="Iteration reduction: preconditioned vs unpreconditioned",
    tags=("2fe", "3fe"),
)
def run_iteration_reduction(opts: SuiteOptions) -> list[BenchmarkResult]:
    maxiter = min(opts.maxiter, 1000)

    if opts.quick:
        problems = [
            ProblemSpec(
                "2-FE 30x40",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40], "n_rows": 2000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 30x40x25",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40, 25], "n_rows": 2000},
                opts.seed,
            ),
        ]
    else:
        problems = [
            ProblemSpec(
                "2-FE 30x40",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40], "n_rows": 2000},
                opts.seed,
            ),
            ProblemSpec(
                "2-FE 50x80",
                "uniform_kfe",
                {"n_levels_per_factor": [50, 80], "n_rows": 5000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 30x40x25",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40, 25], "n_rows": 2000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 50x80x30",
                "uniform_kfe",
                {"n_levels_per_factor": [50, 80, 30], "n_rows": 5000},
                opts.seed,
            ),
        ]

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    configs = [
        SolverConfig(
            "CG(none)",
            CG(tol=opts.tol, maxiter=maxiter),
        ),
        SolverConfig(
            "CG(Schwarz)",
            CG(
                tol=opts.tol,
                maxiter=maxiter,
                preconditioner=AdditiveSchwarz(local_solver=schur),
            ),
        ),
        SolverConfig(
            "GMRES(Mult-Schwarz)",
            GMRES(
                tol=opts.tol,
                maxiter=maxiter,
                preconditioner=MultiplicativeSchwarz(local_solver=schur),
            ),
        ),
    ]

    all_results: list[BenchmarkResult] = []
    for prob in problems:
        gen = get_generator(prob.generator)
        cats, n_levels, y = gen(**prob.params, seed=prob.seed)
        print(f"\nProblem: {prob.name}  (DOFs={sum(n_levels)}, Rows={len(cats[0])})")

        for cfg in configs:
            try:
                result = run_solve(cats, n_levels, y, cfg)
                result.problem = prob.name
                all_results.append(result)
            except Exception as e:
                print(f"  WARNING: {cfg.label} failed: {e}")

    print_table(
        all_results,
        columns=["problem", "config", "iterations", "final_residual", "converged"],
    )

    # Print reduction summary
    print(
        f"\n{'Problem':<18} | {'Solver':<10} | {'Unprec':>7} | {'Prec':>7} | {'Reduction':>9}"
    )
    print("-" * 65)
    for prob in problems:
        label_none = "CG(none)"
        label_prec = "CG(Schwarz)"
        r_none = next(
            (
                r
                for r in all_results
                if r.problem == prob.name and r.config == label_none
            ),
            None,
        )
        r_prec = next(
            (
                r
                for r in all_results
                if r.problem == prob.name and r.config == label_prec
            ),
            None,
        )
        if r_none and r_prec and r_none.iterations > 0:
            reduction = f"{100 * (1 - r_prec.iterations / r_none.iterations):6.1f}%"
        else:
            reduction = "N/A"
        i_none = r_none.iterations if r_none else 0
        i_prec = r_prec.iterations if r_prec else 0
        print(
            f"{prob.name:<18} | {'CG':<10} | {i_none:>7} | {i_prec:>7} | {reduction:>9}"
        )

    return all_results
