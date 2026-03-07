"""AKM panel-data benchmark suites.

Stress-tests Schwarz-preconditioned normal-equation solvers on realistic
matched employer-employee panel structures with the four pathologies that
make real AKM data hard:

1. Power-law (Zipf) firm-size distributions
2. Low worker mobility (sparse bipartite graph)
3. Near-disconnected regional clusters
4. All of the above combined
"""

from __future__ import annotations

from within import CG
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    SchurComplement,
)
from .._framework import (
    BenchmarkResult,
    ProblemSpec,
    SolverConfig,
    SuiteOptions,
    run_problem_set,
    standard_solver_configs,
    suite,
)
from .._table import print_pivot, print_table


# -----------------------------------------------------------------------
# Suite: akm_panel — main comparison across all 4 generators
# -----------------------------------------------------------------------


@suite(
    "akm_panel",
    description="AKM panel pathologies: power-law firms, low mobility, clusters",
    tags=("2fe", "3fe", "akm", "panel"),
)
def run_akm_panel(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec("power_law 10K", "akm_power_law", {}, opts.seed),
            ProblemSpec("low_mobility 10K", "akm_low_mobility", {}, opts.seed),
            ProblemSpec("disconnected 10K", "akm_disconnected", {}, opts.seed),
            ProblemSpec("realistic 10K", "akm_realistic", {}, opts.seed),
        ]
    else:
        problems = [
            # --- 10K workers (quick-tier) ---
            ProblemSpec("power_law 10K", "akm_power_law", {}, opts.seed),
            ProblemSpec("low_mobility 10K", "akm_low_mobility", {}, opts.seed),
            ProblemSpec("disconnected 10K", "akm_disconnected", {}, opts.seed),
            ProblemSpec("realistic 10K", "akm_realistic", {}, opts.seed),
            ProblemSpec("realistic 10K 2FE", "akm_realistic", {"n_fe": 2}, opts.seed),
            # --- 100K workers ---
            ProblemSpec(
                "power_law 100K",
                "akm_power_law",
                {"n_workers": 100_000, "n_firms": 5_000, "n_years": 15},
                opts.seed,
            ),
            ProblemSpec(
                "low_mobility 100K",
                "akm_low_mobility",
                {"n_workers": 100_000, "n_firms": 5_000, "n_years": 15},
                opts.seed,
            ),
            ProblemSpec(
                "disconnected 100K",
                "akm_disconnected",
                {"n_workers": 100_000, "n_firms": 5_000, "n_years": 15},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 100K",
                "akm_realistic",
                {"n_workers": 100_000, "n_firms": 5_000, "n_years": 15},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 100K 2FE",
                "akm_realistic",
                {"n_workers": 100_000, "n_firms": 5_000, "n_years": 15, "n_fe": 2},
                opts.seed,
            ),
            # --- 1M workers ---
            ProblemSpec(
                "realistic 1M",
                "akm_realistic",
                {"n_workers": 1_000_000, "n_firms": 50_000, "n_years": 20},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 1M 2FE",
                "akm_realistic",
                {"n_workers": 1_000_000, "n_firms": 50_000, "n_years": 20, "n_fe": 2},
                opts.seed,
            ),
        ]

    configs = standard_solver_configs(opts)
    all_results = run_problem_set(problems, configs)
    print_table(all_results)
    print("\n")
    print_pivot(all_results)
    return all_results


# -----------------------------------------------------------------------
# Suite: akm_scaling — scaling behavior of akm_realistic
# -----------------------------------------------------------------------


@suite(
    "akm_scaling",
    description="AKM realistic scaling: wall-clock vs problem size",
    tags=("akm", "scaling"),
)
def run_akm_scaling(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec(
                "realistic 10K",
                "akm_realistic",
                {"n_workers": 10_000, "n_firms": 500, "n_years": 10},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 50K",
                "akm_realistic",
                {"n_workers": 50_000, "n_firms": 2_500, "n_years": 12},
                opts.seed,
            ),
        ]
    else:
        problems = [
            ProblemSpec(
                "realistic 10K",
                "akm_realistic",
                {"n_workers": 10_000, "n_firms": 500, "n_years": 10},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 50K",
                "akm_realistic",
                {"n_workers": 50_000, "n_firms": 2_500, "n_years": 12},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 100K",
                "akm_realistic",
                {"n_workers": 100_000, "n_firms": 5_000, "n_years": 15},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 500K",
                "akm_realistic",
                {"n_workers": 500_000, "n_firms": 25_000, "n_years": 18},
                opts.seed,
            ),
            ProblemSpec(
                "realistic 1M",
                "akm_realistic",
                {"n_workers": 1_000_000, "n_firms": 50_000, "n_years": 20},
                opts.seed,
            ),
        ]

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    scaling_configs = [
        SolverConfig(
            "CG(Schwarz)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=schur),
            ),
        ),
    ]

    all_results = run_problem_set(problems, scaling_configs)
    print_table(all_results)
    print("\n")
    print_pivot(all_results)
    return all_results
