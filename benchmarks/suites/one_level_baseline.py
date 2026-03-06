"""One-level Schwarz baseline on 3-FE problems."""

from __future__ import annotations

from within import (
    ApproxCholConfig,
    CG,
    OneLevelSchwarz,
)

from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solvers import run_solve
from .._table import print_pivot, print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig


@suite(
    "one_level_baseline",
    description="One-level Schwarz baseline on large 3-FE problems",
    tags=("3fe", "baseline"),
)
def run_one_level_baseline(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec(
                "Sparse 100^3 e=3",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Clustered 100^3",
                "clustered_3fe",
                {
                    "n_levels": (100, 100, 100),
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec("Barbell 3fe 250", "barbell_3fe", {"n_levels": 250}, opts.seed),
            ProblemSpec("Chain 3fe 250", "chain_3fe", {"n_levels": 250}, opts.seed),
        ]
    else:
        problems = [
            ProblemSpec(
                "Sparse 100^3 e=3",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse 200^3 e=3",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse 100^3 e=5",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 5},
                opts.seed,
            ),
            ProblemSpec(
                "Clustered 100^3",
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
                "Clustered 200^3",
                "clustered_3fe",
                {
                    "n_levels": (200, 200, 200),
                    "n_clusters": 10,
                    "obs_per_cluster": 500,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec("Barbell 3fe 250", "barbell_3fe", {"n_levels": 250}, opts.seed),
            ProblemSpec("Barbell 3fe 500", "barbell_3fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("Chain 3fe 250", "chain_3fe", {"n_levels": 250}, opts.seed),
            ProblemSpec("Chain 3fe 500", "chain_3fe", {"n_levels": 500}, opts.seed),
            ProblemSpec(
                "Imbalanced 200^3",
                "imbalanced_3fe",
                {"n_levels": (200, 200, 200), "n_rows": 30000},
                opts.seed,
            ),
            ProblemSpec(
                "AKM power-law",
                "akm_power_law",
                {"n_workers": 5000, "n_firms": 300, "n_years": 10},
                opts.seed,
            ),
            ProblemSpec(
                "AKM disconnected",
                "akm_disconnected",
                {"n_workers": 5000, "n_firms": 300, "n_years": 10, "n_clusters": 5},
                opts.seed,
            ),
        ]

    smoother = ApproxCholConfig(seed=opts.seed)

    configs = [
        SolverConfig(
            "1L Schwarz",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=OneLevelSchwarz(smoother=smoother),
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

    # Also show setup+solve time pivots
    print()
    print_pivot(all_results, value="setup_time")
    print()
    print_pivot(all_results, value="solve_time")

    return all_results
