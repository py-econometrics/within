"""AC vs AC2 local-solver comparison."""

from __future__ import annotations

from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solver_presets import cg_solver_config, gmres_solver_config
from .._solvers import run_solve
from .._table import print_pivot, print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig


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
        # One-level additive
        SolverConfig(
            "CG(1L, AC)",
            cg_solver_config(opts, split=1).config,
        ),
        SolverConfig(
            "CG(1L, AC2)",
            cg_solver_config(opts, split=2).config,
        ),
        SolverConfig(
            "GMRES(M1L, AC)",
            gmres_solver_config(opts, split=1).config,
        ),
        SolverConfig(
            "GMRES(M1L, AC2)",
            gmres_solver_config(opts, split=2).config,
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
