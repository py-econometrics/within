"""Fixest-style DGP benchmark suite.

Runs the 3-FE "difficult" (sequential/block firm assignment) variant of the
fixest panel DGP through our solver pipeline, scaling from 100K to 5M
observations.
"""

from __future__ import annotations

import os

from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    MultiplicativeSchwarz,
    SchurComplement,
)
from .._problems import get_generator
from .._framework import (
    BenchmarkResult,
    SolverConfig,
    SuiteOptions,
    benchmark_cg,
    benchmark_gmres,
    make_additive_schwarz,
    run_solve,
    suite,
)
from .._table import print_pivot, print_table


@suite(
    "fixest_comparison",
    description="Fixest-style 3FE difficult panel DGP up to 320M obs",
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
            320_000_000,
        ],
    )
    max_obs = int(
        os.environ.get("WITHIN_BENCH_FIXEST_MAX_OBS", str(max(n_obs_list, default=0)))
    )
    n_obs_list = [n_obs for n_obs in n_obs_list if n_obs <= max_obs]

    fixest_variants = (
        os.environ.get(
            "WITHIN_BENCH_FIXEST_VARIANTS",
            "both" if opts.profile == "full" else "dense",
        )
        .strip()
        .lower()
    )

    solver_configs = [
        SolverConfig(
            "CG(Schwarz)",
            benchmark_cg(opts),
            preconditioner=make_additive_schwarz(
                local_solver=SchurComplement(
                    approx_chol=ApproxCholConfig(seed=0, split=8),
                    approx_schur=None,
                )
            ),
        ),
    ]
    if fixest_variants == "both":
        solver_configs.append(
            SolverConfig(
                "CG(Schwarz)-sparse",
                benchmark_cg(opts),
                preconditioner=make_additive_schwarz(
                    local_solver=SchurComplement(
                        approx_chol=ApproxCholConfig(seed=0, split=2),
                        approx_schur=ApproxSchurConfig(seed=0, split=2),
                    )
                ),
            )
        )
    elif fixest_variants != "dense":
        raise ValueError("WITHIN_BENCH_FIXEST_VARIANTS must be one of: dense, both")

    if opts.profile != "full":
        solver_configs.append(
            SolverConfig(
                "GMRES(Mult-Schwarz)",
                benchmark_gmres(opts),
                preconditioner=MultiplicativeSchwarz(
                    local_solver=SchurComplement(
                        approx_chol=ApproxCholConfig(seed=0, split=8),
                        approx_schur=None,
                    )
                ),
            )
        )

    gen = get_generator("fixest_dgp")
    all_results: list[BenchmarkResult] = []

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
                all_results.append(result)
            except Exception as e:
                print(f"    WARNING: {cfg.label} failed: {e}")

    print_table(all_results)
    print_table(
        all_results,
        columns=["config", "setup_time", "solve_time", "iterations", "ms_per_iter"],
        title="Per-iteration cost",
    )
    print("\n")
    print_pivot(all_results)
    return all_results
