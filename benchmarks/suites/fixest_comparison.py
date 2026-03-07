"""Fixest-style DGP benchmark suite.

Runs the 3-FE "difficult" (sequential/block firm assignment) variant of the
fixest panel DGP through our solver pipeline, scaling from 100K to 5M
observations.
"""

from __future__ import annotations

from within import CG
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    SchurComplement,
    ApproxSchurConfig,
)
from .._problems import get_generator
from .._framework import BenchmarkResult, SolverConfig, SuiteOptions, run_solve, suite
from .._table import print_pivot, print_table


@suite(
    "fixest_comparison",
    description="Fixest-style 3FE difficult panel DGP up to 320M obs",
    tags=("3fe", "fixest", "scaling", "difficult"),
)
def run_fixest_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        n_obs_list = [100_000, 500_000]
    else:
        n_obs_list = [
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
        ]

    solver_configs = [
        SolverConfig(
            "CG(Schwarz)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(
                    local_solver=SchurComplement(
                        approx_chol=ApproxCholConfig(seed=0, split=8),
                        approx_schur=None,
                    )
                ),
            ),
        ),
        SolverConfig(
            "CG(Schwarz)-sparse",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(
                    local_solver=SchurComplement(
                        approx_chol=ApproxCholConfig(seed=0, split=2),
                        approx_schur=ApproxSchurConfig(seed=0, split=2),
                    )
                ),
            ),
        ),
    ]

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
                result = run_solve(cats, n_levels, y, cfg)
                result.problem = name
                all_results.append(result)
            except Exception as e:
                print(f"    WARNING: {cfg.label} failed: {e}")

    print_table(all_results)
    print("\n")
    print_pivot(all_results)
    return all_results
