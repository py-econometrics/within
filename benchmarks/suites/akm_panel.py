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

from within import CG, GMRES
from within._within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    MultiplicativeSchwarz,
    SchurComplement,
)
from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solvers import run_solve
from .._table import print_pivot, print_table
from .._types import BenchmarkResult, SolverConfig


def _solver_configs(opts: SuiteOptions) -> list[SolverConfig]:
    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    return [
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
            ("power_law 10K", "akm_power_law", {}),
            ("low_mobility 10K", "akm_low_mobility", {}),
            ("disconnected 10K", "akm_disconnected", {}),
            ("realistic 10K", "akm_realistic", {}),
        ]
        solver_sets = [_solver_configs(opts)] * len(problems)
    else:
        problems = [
            # --- 10K workers (quick-tier) ---
            ("power_law 10K", "akm_power_law", {}),
            ("low_mobility 10K", "akm_low_mobility", {}),
            ("disconnected 10K", "akm_disconnected", {}),
            ("realistic 10K", "akm_realistic", {}),
            ("realistic 10K 2FE", "akm_realistic", {"n_fe": 2}),
            # --- 100K workers ---
            (
                "power_law 100K",
                "akm_power_law",
                {
                    "n_workers": 100_000,
                    "n_firms": 5_000,
                    "n_years": 15,
                },
            ),
            (
                "low_mobility 100K",
                "akm_low_mobility",
                {
                    "n_workers": 100_000,
                    "n_firms": 5_000,
                    "n_years": 15,
                },
            ),
            (
                "disconnected 100K",
                "akm_disconnected",
                {
                    "n_workers": 100_000,
                    "n_firms": 5_000,
                    "n_years": 15,
                },
            ),
            (
                "realistic 100K",
                "akm_realistic",
                {
                    "n_workers": 100_000,
                    "n_firms": 5_000,
                    "n_years": 15,
                },
            ),
            (
                "realistic 100K 2FE",
                "akm_realistic",
                {
                    "n_workers": 100_000,
                    "n_firms": 5_000,
                    "n_years": 15,
                    "n_fe": 2,
                },
            ),
            # --- 1M workers ---
            (
                "realistic 1M",
                "akm_realistic",
                {
                    "n_workers": 1_000_000,
                    "n_firms": 50_000,
                    "n_years": 20,
                },
            ),
            (
                "realistic 1M 2FE",
                "akm_realistic",
                {
                    "n_workers": 1_000_000,
                    "n_firms": 50_000,
                    "n_years": 20,
                    "n_fe": 2,
                },
            ),
        ]
        solver_sets = [_solver_configs(opts)] * len(problems)

    all_results: list[BenchmarkResult] = []
    for (name, gen_key, params), solvers in zip(problems, solver_sets):
        gen = get_generator(gen_key)
        cats, n_levels, y = gen(**params, seed=opts.seed)
        print(f"\n  {name}: DOFs={sum(n_levels):,}, Rows={len(cats[0]):,}")

        for cfg in solvers:
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


# -----------------------------------------------------------------------
# Suite: akm_scaling — scaling behavior of akm_realistic
# -----------------------------------------------------------------------


@suite(
    "akm_scaling",
    description="AKM realistic scaling: wall-clock vs problem size",
    tags=("akm", "scaling"),
)
def run_akm_scaling(opts: SuiteOptions) -> list[BenchmarkResult]:
    gen = get_generator("akm_realistic")

    if opts.quick:
        sizes = [
            ("10K", {"n_workers": 10_000, "n_firms": 500, "n_years": 10}),
            ("50K", {"n_workers": 50_000, "n_firms": 2_500, "n_years": 12}),
        ]
    else:
        sizes = [
            ("10K", {"n_workers": 10_000, "n_firms": 500, "n_years": 10}),
            ("50K", {"n_workers": 50_000, "n_firms": 2_500, "n_years": 12}),
            ("100K", {"n_workers": 100_000, "n_firms": 5_000, "n_years": 15}),
            ("500K", {"n_workers": 500_000, "n_firms": 25_000, "n_years": 18}),
            ("1M", {"n_workers": 1_000_000, "n_firms": 50_000, "n_years": 20}),
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

    all_results: list[BenchmarkResult] = []
    for label, params in sizes:
        cats, n_levels, y = gen(**params, seed=opts.seed)
        print(f"\n  realistic {label}: DOFs={sum(n_levels):,}, Rows={len(cats[0]):,}")

        for cfg in scaling_configs:
            try:
                result = run_solve(cats, n_levels, y, cfg)
                result.problem = f"realistic {label}"
                all_results.append(result)
            except Exception as e:
                print(f"    WARNING: {cfg.label} failed: {e}")

    print_table(all_results)
    print("\n")
    print_pivot(all_results)
    return all_results
