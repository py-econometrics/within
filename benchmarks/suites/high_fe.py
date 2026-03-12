"""High fixed-effects suites (4-6 FE).

Tests solver behaviour as the number of fixed effects grows beyond 3,
which multiplies the number of factor-pair subdomains and stresses
domain decomposition.
"""

from __future__ import annotations

from within._within import (
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
    benchmark_cg,
    benchmark_gmres,
    make_additive_schwarz,
    run_problem_set,
    suite,
)
from .._table import print_pivot, print_table


@suite(
    "high_fe",
    description="4-6 FE: random, sparse, chain, imbalanced problems",
    tags=("4fe", "5fe", "6fe", "high_fe"),
)
def run_high_fe(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = opts.select(
        smoke=[
            ProblemSpec(
                "random 4fe 30^4",
                "random_kfe",
                {"k": 4, "n_levels_per_factor": [30, 30, 30, 30], "n_rows": 5000},
                opts.seed,
            ),
            ProblemSpec(
                "sparse 4fe 30^4",
                "sparse_kfe",
                {"k": 4, "n_levels": 30, "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "random 5fe 20^5",
                "random_kfe",
                {"k": 5, "n_levels_per_factor": [20, 20, 20, 20, 20], "n_rows": 5000},
                opts.seed,
            ),
            ProblemSpec(
                "random 6fe 15^6",
                "random_kfe",
                {
                    "k": 6,
                    "n_levels_per_factor": [15, 15, 15, 15, 15, 15],
                    "n_rows": 5000,
                },
                opts.seed,
            ),
        ],
        iterate=[
            ProblemSpec(
                "random 4fe asym",
                "random_kfe",
                {"k": 4, "n_levels_per_factor": [200, 100, 50, 20], "n_rows": 20000},
                opts.seed,
            ),
            ProblemSpec(
                "sparse 4fe 50^4",
                "sparse_kfe",
                {"k": 4, "n_levels": 50, "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "chain 4fe 50^4", "chain_kfe", {"k": 4, "n_levels": 50}, opts.seed
            ),
            ProblemSpec(
                "random 5fe asym",
                "random_kfe",
                {"k": 5, "n_levels_per_factor": [100, 50, 30, 20, 10], "n_rows": 20000},
                opts.seed,
            ),
            ProblemSpec(
                "sparse 5fe 30^5",
                "sparse_kfe",
                {"k": 5, "n_levels": 30, "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "random 6fe asym",
                "random_kfe",
                {
                    "k": 6,
                    "n_levels_per_factor": [80, 40, 30, 20, 15, 10],
                    "n_rows": 20000,
                },
                opts.seed,
            ),
            ProblemSpec(
                "sparse 6fe 20^6",
                "sparse_kfe",
                {"k": 6, "n_levels": 20, "edges_per_level": 3},
                opts.seed,
            ),
        ],
        full=[
            # 4-FE
            ProblemSpec(
                "random 4fe 50^4",
                "random_kfe",
                {"k": 4, "n_levels_per_factor": [50, 50, 50, 50], "n_rows": 20000},
                opts.seed,
            ),
            ProblemSpec(
                "random 4fe asym",
                "random_kfe",
                {"k": 4, "n_levels_per_factor": [200, 100, 50, 20], "n_rows": 20000},
                opts.seed,
            ),
            ProblemSpec(
                "sparse 4fe 50^4",
                "sparse_kfe",
                {"k": 4, "n_levels": 50, "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "chain 4fe 50^4", "chain_kfe", {"k": 4, "n_levels": 50}, opts.seed
            ),
            ProblemSpec(
                "imbal 4fe 50^4",
                "imbalanced_kfe",
                {"k": 4, "n_levels_per_factor": [50, 50, 50, 50], "n_rows": 20000},
                opts.seed,
            ),
            # 5-FE
            ProblemSpec(
                "random 5fe 30^5",
                "random_kfe",
                {"k": 5, "n_levels_per_factor": [30, 30, 30, 30, 30], "n_rows": 20000},
                opts.seed,
            ),
            ProblemSpec(
                "random 5fe asym",
                "random_kfe",
                {"k": 5, "n_levels_per_factor": [100, 50, 30, 20, 10], "n_rows": 20000},
                opts.seed,
            ),
            ProblemSpec(
                "sparse 5fe 30^5",
                "sparse_kfe",
                {"k": 5, "n_levels": 30, "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "chain 5fe 30^5", "chain_kfe", {"k": 5, "n_levels": 30}, opts.seed
            ),
            # 6-FE
            ProblemSpec(
                "random 6fe 20^6",
                "random_kfe",
                {
                    "k": 6,
                    "n_levels_per_factor": [20, 20, 20, 20, 20, 20],
                    "n_rows": 20000,
                },
                opts.seed,
            ),
            ProblemSpec(
                "random 6fe asym",
                "random_kfe",
                {
                    "k": 6,
                    "n_levels_per_factor": [80, 40, 30, 20, 15, 10],
                    "n_rows": 20000,
                },
                opts.seed,
            ),
            ProblemSpec(
                "sparse 6fe 20^6",
                "sparse_kfe",
                {"k": 6, "n_levels": 20, "edges_per_level": 3},
                opts.seed,
            ),
        ],
    )

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    configs = [
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
    results = run_problem_set(problems, configs, opts)
    print_table(results)
    print("\n")
    print_pivot(results)
    return results


@suite(
    "high_fe_scaling",
    description="Scaling behaviour as k (number of FE) grows from 2 to 6",
    tags=("scaling", "high_fe"),
)
def run_high_fe_scaling(opts: SuiteOptions) -> list[BenchmarkResult]:
    """Fix total DOFs ~ 200 and rows ~ 10K, vary k from 2 to 6."""
    problems = opts.select(
        smoke=[
            ProblemSpec(
                "2-FE 50x50",
                "random_kfe",
                {"k": 2, "n_levels_per_factor": [50, 50], "n_rows": 5000},
                opts.seed,
            ),
            ProblemSpec(
                "4-FE 25^4",
                "random_kfe",
                {"k": 4, "n_levels_per_factor": [25, 25, 25, 25], "n_rows": 5000},
                opts.seed,
            ),
            ProblemSpec(
                "6-FE 18^6",
                "random_kfe",
                {
                    "k": 6,
                    "n_levels_per_factor": [18, 18, 18, 18, 18, 18],
                    "n_rows": 5000,
                },
                opts.seed,
            ),
        ],
        iterate=[
            ProblemSpec(
                "2-FE 100x100",
                "random_kfe",
                {"k": 2, "n_levels_per_factor": [100, 100], "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 65^3",
                "random_kfe",
                {"k": 3, "n_levels_per_factor": [65, 65, 65], "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "4-FE 50^4",
                "random_kfe",
                {"k": 4, "n_levels_per_factor": [50, 50, 50, 50], "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "6-FE 33^6",
                "random_kfe",
                {
                    "k": 6,
                    "n_levels_per_factor": [33, 33, 33, 33, 33, 33],
                    "n_rows": 10000,
                },
                opts.seed,
            ),
        ],
        full=[
            ProblemSpec(
                "2-FE 100x100",
                "random_kfe",
                {"k": 2, "n_levels_per_factor": [100, 100], "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 65^3",
                "random_kfe",
                {"k": 3, "n_levels_per_factor": [65, 65, 65], "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "4-FE 50^4",
                "random_kfe",
                {"k": 4, "n_levels_per_factor": [50, 50, 50, 50], "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "5-FE 40^5",
                "random_kfe",
                {"k": 5, "n_levels_per_factor": [40, 40, 40, 40, 40], "n_rows": 10000},
                opts.seed,
            ),
            ProblemSpec(
                "6-FE 33^6",
                "random_kfe",
                {
                    "k": 6,
                    "n_levels_per_factor": [33, 33, 33, 33, 33, 33],
                    "n_rows": 10000,
                },
                opts.seed,
            ),
        ],
    )

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    configs = [
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
    results = run_problem_set(problems, configs, opts)
    print_table(results)
    print("\n")
    print_pivot(results)
    return results
