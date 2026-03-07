"""AC vs AC2 local-solver comparison.

Two suites:
- ``ac_comparison`` — CG + GMRES with AC vs AC2 on mixed topologies
- ``graph_backend_comparison`` — CG-only AC vs AC2 on large-scale 2-FE and 3-FE topologies
"""

from __future__ import annotations

from within import CG, GMRES
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    MultiplicativeSchwarz,
    SchurComplement,
)
from .._framework import (
    BenchmarkResult,
    ProblemSpec,
    SolverConfig,
    SuiteOptions,
    run_problem_set,
    suite,
)
from .._table import print_pivot, print_table


def _schur(seed: int, split: int) -> SchurComplement:
    return SchurComplement(
        approx_chol=ApproxCholConfig(seed=seed, split=split),
        approx_schur=ApproxSchurConfig(seed=seed),
    )


@suite(
    "ac_comparison",
    description="AC vs AC2 local solver across preconditioner types",
    tags=("local_solver", "precond"),
)
def run_ac_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec(
                "Sparse 50^3 3e",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec("Chain 100 2fe", "chain_2fe", {"n_levels": 100}, opts.seed),
        ]
    else:
        problems = [
            ProblemSpec("Chain 200 2fe", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec(
                "Expander 100 2fe",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse 50^3 3e",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse 100^3 3e",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse 100^3 5e",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 5},
                opts.seed,
            ),
            ProblemSpec(
                "Imbalanced 100^3",
                "imbalanced_3fe",
                {"n_levels": (100, 100, 100), "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "AKM Power-Law",
                "akm_power_law",
                {"n_workers": 5000, "n_firms": 200, "n_years": 10},
                opts.seed,
            ),
        ]

    configs = [
        SolverConfig(
            "CG(1L, AC)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=_schur(opts.seed, 1)),
            ),
        ),
        SolverConfig(
            "CG(1L, AC2)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=_schur(opts.seed, 2)),
            ),
        ),
        SolverConfig(
            "GMRES(M1L, AC)",
            GMRES(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=MultiplicativeSchwarz(local_solver=_schur(opts.seed, 1)),
            ),
        ),
        SolverConfig(
            "GMRES(M1L, AC2)",
            GMRES(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=MultiplicativeSchwarz(local_solver=_schur(opts.seed, 2)),
            ),
        ),
    ]

    all_results = run_problem_set(problems, configs)
    print_table(
        all_results,
        columns=[
            "problem",
            "config",
            "setup_time",
            "solve_time",
            "iterations",
            "final_residual",
            "converged",
        ],
    )
    print("\n--- Iterations pivot ---")
    print_pivot(all_results)
    print("\n--- Setup time pivot ---")
    print_pivot(all_results, value="setup_time")
    print("\n--- Solve time pivot ---")
    print_pivot(all_results, value="solve_time")
    return all_results


@suite(
    "graph_backend_comparison",
    description="ApproxChol variant comparison across large-scale topologies",
    tags=("2fe", "3fe", "ac"),
)
def run_graph_backend_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec("chain 100 2fe", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("chain 500 2fe", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("star 100 2fe", "star_2fe", {"n_levels": 100}, opts.seed),
        ]
    else:
        problems = [
            ProblemSpec("chain 500 2fe", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("chain 2000 2fe", "chain_2fe", {"n_levels": 2000}, opts.seed),
            ProblemSpec("chain 5000 2fe", "chain_2fe", {"n_levels": 5000}, opts.seed),
            ProblemSpec("chain 10000 2fe", "chain_2fe", {"n_levels": 10000}, opts.seed),
            ProblemSpec("star 500 2fe", "star_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("star 2000 2fe", "star_2fe", {"n_levels": 2000}, opts.seed),
            ProblemSpec("star 5000 2fe", "star_2fe", {"n_levels": 5000}, opts.seed),
            ProblemSpec(
                "expander 500 d=3",
                "expander_2fe",
                {"n_levels": 500, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 2000 d=3",
                "expander_2fe",
                {"n_levels": 2000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 5000 d=3",
                "expander_2fe",
                {"n_levels": 5000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 500 d=10",
                "expander_2fe",
                {"n_levels": 500, "degree": 10},
                opts.seed,
            ),
            ProblemSpec(
                "expander 2000 d=10",
                "expander_2fe",
                {"n_levels": 2000, "degree": 10},
                opts.seed,
            ),
            ProblemSpec("barbell 500 2fe", "barbell_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec(
                "barbell 2000 2fe", "barbell_2fe", {"n_levels": 2000}, opts.seed
            ),
            ProblemSpec("grid 50x50 2fe", "grid_2fe", {"n_side": 50}, opts.seed),
            ProblemSpec("grid 100x100 2fe", "grid_2fe", {"n_side": 100}, opts.seed),
            ProblemSpec(
                "sparse 200^3 3fe",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "sparse 500^3 3fe",
                "sparse_3fe",
                {"n_levels": (500, 500, 500), "edges_per_level": 3},
                opts.seed,
            ),
        ]

    configs = [
        SolverConfig(
            "ac",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=_schur(opts.seed, 1)),
            ),
        ),
        SolverConfig(
            "ac2",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=_schur(opts.seed, 2)),
            ),
        ),
    ]

    all_results = run_problem_set(problems, configs)
    print_table(all_results)
    print("\nSetup time pivot:")
    print_pivot(all_results, value="setup_time")
    print("\nIteration count pivot:")
    print_pivot(all_results, value="iterations")
    return all_results
