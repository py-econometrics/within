"""Preconditioner comparison suites."""

from __future__ import annotations

from within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    CG,
    GMRES,
    MultiplicativeSchwarz,
    SchurComplement,
)
from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solvers import run_solve
from .._table import print_pivot, print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig


@suite(
    "preconditioners_3fe",
    description="CG(additive Schwarz) vs GMRES(multiplicative Schwarz) on 3-FE problems",
    tags=("3fe", "precond"),
)
def run_preconditioners_3fe(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec(
                "Sparse (50^3, 3e)",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Imbalanced (50^3)",
                "imbalanced_3fe",
                {"n_levels": (50, 50, 50), "n_rows": 5000},
                opts.seed,
            ),
        ]
    else:
        problems = [
            ProblemSpec(
                "Sparse (50^3, 3e)",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse (50^3, 5e)",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 5},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse (100^3, 3e)",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse (100^3, 5e)",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 5},
                opts.seed,
            ),
            ProblemSpec(
                "V.Sparse (200^3, 2e)",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 2},
                opts.seed,
            ),
            ProblemSpec(
                "V.Sparse (200^3, 3e)",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Clustered (100^3)",
                "clustered_3fe",
                {
                    "n_levels": (100, 100, 100),
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "Imbalanced (100^3)",
                "imbalanced_3fe",
                {"n_levels": (100, 100, 100), "n_rows": 10000},
                opts.seed,
            ),
        ]

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    configs = [
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

    print_table(all_results)
    print("\n")
    print_pivot(all_results)
    return all_results


@suite(
    "preconditioner_comparison",
    description="Unpreconditioned CG vs Schwarz-preconditioned CG/GMRES",
    tags=("2fe", "3fe", "precond"),
)
def run_preconditioner_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec("chain 50 2fe", "chain_2fe", {"n_levels": 50}, opts.seed),
            ProblemSpec(
                "sparse 50^3 3fe",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
        ]
    else:
        problems = [
            ProblemSpec("chain 100 2fe", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("chain 200 2fe", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec(
                "expander 100 2fe",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
            ProblemSpec("barbell 100 2fe", "barbell_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec(
                "sparse 100^3 3fe",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec("chain 100 3fe", "chain_3fe", {"n_levels": 100}, opts.seed),
        ]

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    configs = [
        SolverConfig(
            "CG(none)",
            CG(tol=opts.tol, maxiter=opts.maxiter),
        ),
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

    print_table(all_results)
    print("\n")
    print_pivot(all_results)
    return all_results
