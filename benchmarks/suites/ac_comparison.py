"""AC vs AC2 local-solver comparison.

Two suites:
- ``ac_comparison`` — AC vs AC2 on mixed 2-FE / 3-FE topologies
- ``graph_backend_comparison`` — AC vs AC2 on large-scale chain / star /
  expander / barbell / grid / sparse topologies (scaling sweep)

The ``split_merge`` parameter on ``ApproxCholConfig`` controls how many
copies each star edge is split into before clique-tree sampling.
``split_merge=None`` (AC) is the standard sparser approximation;
``split_merge=2`` (AC2) is denser and gives a better Schur approximation
at the cost of more fill-in.
"""

from __future__ import annotations

from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    LocalSolverConfig,
)
from .._framework import (
    BenchmarkResult,
    ProblemSpec,
    SolverConfig,
    SuiteOptions,
    benchmark_lsmr,
    make_additive_schwarz,
    run_problem_set,
    suite,
)
from .._table import print_pivot, print_table


def _schur(seed: int, split_merge: int | None) -> LocalSolverConfig:
    return LocalSolverConfig(
        approx_chol=ApproxCholConfig(seed=seed, split_merge=split_merge),
        approx_schur=ApproxSchurConfig(seed=seed),
    )


@suite(
    "ac_comparison",
    description="AC vs AC2 local solver on mixed 2-FE / 3-FE topologies",
    tags=("local_solver", "precond"),
)
def run_ac_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = opts.select(
        smoke=[
            ProblemSpec(
                "Sparse 50^3 3e",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec("Chain 100 2fe", "chain_2fe", {"n_levels": 100}, opts.seed),
        ],
        iterate=[
            ProblemSpec("Chain 200 2fe", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec(
                "Expander 100 2fe",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse 100^3 3e",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
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
        ],
        full=[
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
        ],
    )

    configs = [
        SolverConfig(
            "LSMR(AC)",
            benchmark_lsmr(opts),
            preconditioner=make_additive_schwarz(local_solver=_schur(opts.seed, None)),
        ),
        SolverConfig(
            "LSMR(AC2)",
            benchmark_lsmr(opts),
            preconditioner=make_additive_schwarz(local_solver=_schur(opts.seed, 2)),
        ),
    ]

    results = run_problem_set(problems, configs, opts)
    print_table(results)
    print("\n--- Iterations pivot ---")
    print_pivot(results)
    print("\n--- Setup time pivot ---")
    print_pivot(results, value="setup_time")
    print("\n--- Solve time pivot ---")
    print_pivot(results, value="solve_time")
    return results


@suite(
    "graph_backend_comparison",
    description="AC vs AC2 across large-scale 2-FE / 3-FE graph topologies",
    tags=("2fe", "3fe", "ac"),
)
def run_graph_backend_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    maxiter = max(opts.maxiter, 6000)
    problems = opts.select(
        smoke=[
            ProblemSpec("chain 100 2fe", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("chain 500 2fe", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("star 100 2fe", "star_2fe", {"n_levels": 100}, opts.seed),
        ],
        iterate=[
            ProblemSpec("chain 500 2fe", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("chain 2000 2fe", "chain_2fe", {"n_levels": 2000}, opts.seed),
            ProblemSpec("star 500 2fe", "star_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("barbell 500 2fe", "barbell_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec(
                "expander 500 d=3",
                "expander_2fe",
                {"n_levels": 500, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 500 d=10",
                "expander_2fe",
                {"n_levels": 500, "degree": 10},
                opts.seed,
            ),
            ProblemSpec("grid 50x50 2fe", "grid_2fe", {"n_side": 50}, opts.seed),
            ProblemSpec(
                "sparse 200^3 3fe",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 3},
                opts.seed,
            ),
        ],
        full=[
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
        ],
    )

    configs = [
        SolverConfig(
            "ac",
            benchmark_lsmr(opts, maxiter=maxiter),
            preconditioner=make_additive_schwarz(local_solver=_schur(opts.seed, None)),
        ),
        SolverConfig(
            "ac2",
            benchmark_lsmr(opts, maxiter=maxiter),
            preconditioner=make_additive_schwarz(local_solver=_schur(opts.seed, 2)),
        ),
    ]

    results = run_problem_set(problems, configs, opts)
    print_table(results)
    print("\nSetup time pivot:")
    print_pivot(results, value="setup_time")
    print("\nIteration count pivot:")
    print_pivot(results, value="iterations")
    return results
