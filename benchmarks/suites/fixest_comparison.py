"""Fixest-style 3-FE difficult panel DGP scaling suite.

Runs the 3-FE "difficult" (sequential/block firm assignment) variant of the
fixest panel DGP through LSMR Schwarz, comparing exact vs approximate Schur
complement on the local solver. Scales from 100K to 160M observations.
"""

from __future__ import annotations

from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    Schur,
    LocalSolverConfig,
)
from .._framework import (
    BenchmarkResult,
    SolverConfig,
    SuiteOptions,
    benchmark_lsmr,
    make_additive_schwarz,
    run_solve,
    suite,
)
from .._problems import get_generator
from .._table import print_pivot, print_table


@suite(
    "fixest_comparison",
    description="Fixest-style 3FE difficult panel DGP, exact vs approx Schur",
    tags=("3fe", "fixest", "scaling", "difficult"),
)
def run_fixest_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    n_obs_list = opts.select(
        smoke=[100_000, 500_000],
        iterate=[500_000, 2_000_000, 5_000_000, 20_000_000],
        full=[
            100_000,
            500_000,
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
            10_000_000,
            20_000_000,
            40_000_000,
            80_000_000,
            160_000_000,
        ],
    )

    exact_schur = LocalSolverConfig(
        approx_chol=ApproxCholConfig(seed=0, split_merge=8),
        schur=Schur.exact(),
    )
    approx_schur = LocalSolverConfig(
        approx_chol=ApproxCholConfig(seed=0, split_merge=8),
        schur=Schur.approximate(ApproxSchurConfig(seed=0)),
    )
    # "2-2": both densification knobs on — split_merge=2 (AC2 factorization)
    # and split=2 (denser Schur clique-tree sampling).
    schur_2_2 = LocalSolverConfig(
        approx_chol=ApproxCholConfig(seed=0, split_merge=2),
        schur=Schur.approximate(ApproxSchurConfig(seed=0, split=2)),
    )

    solver_configs = [
        SolverConfig(
            "LSMR(exact)",
            benchmark_lsmr(opts),
            preconditioner=make_additive_schwarz(local_solver=exact_schur),
        ),
        SolverConfig(
            "LSMR(approx)",
            benchmark_lsmr(opts),
            preconditioner=make_additive_schwarz(local_solver=approx_schur),
        ),
        SolverConfig(
            "LSMR(2-2)",
            benchmark_lsmr(opts),
            preconditioner=make_additive_schwarz(local_solver=schur_2_2),
        ),
    ]

    gen = get_generator("fixest_dgp")
    results: list[BenchmarkResult] = []

    for n_obs in n_obs_list:
        name = f"n={n_obs:,} difficult 3FE"
        cats, n_levels, y = gen(
            n_obs=n_obs, dgp_type="difficult", n_fe=3, seed=opts.seed
        )
        print(f"\n  {name}: DOFs={sum(n_levels)}, Rows={len(cats[0])}")

        for cfg in solver_configs:
            try:
                result = run_solve(cats, n_levels, y, cfg, opts)
                result.problem = name
                results.append(result)
            except Exception as e:
                print(f"    WARNING: {cfg.label} failed: {e}")

    print_table(results)
    print_table(
        results,
        columns=["config", "setup_time", "solve_time", "iterations", "ms_per_iter"],
        title="Per-iteration cost",
    )
    print("\n")
    print_pivot(results)
    return results
