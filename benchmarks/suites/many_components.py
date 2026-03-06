"""Many connected-components suites.

Tests solver behaviour on problems with many (near-)disconnected
components in the bipartite factor graph.  These arise naturally in
employer-employee data with regional labour markets and low mobility.
"""

from __future__ import annotations

from within import (
    ApproxCholConfig,
    CG,
    GMRES,
    LSMR,
    MultiplicativeOneLevelSchwarz,
    OneLevelSchwarz,
)

from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solvers import run_solve
from .._table import print_pivot, print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig


def _make_configs(opts: SuiteOptions) -> list[SolverConfig]:
    return [
        SolverConfig("LSMR(diag)", LSMR(tol=opts.tol, maxiter=opts.maxiter)),
        SolverConfig(
            "CG(Schwarz)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=OneLevelSchwarz(
                    smoother=ApproxCholConfig(seed=opts.seed)
                ),
            ),
        ),
        SolverConfig(
            "GMRES(Mult-Schwarz)",
            GMRES(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=MultiplicativeOneLevelSchwarz(
                    smoother=ApproxCholConfig(seed=opts.seed)
                ),
            ),
        ),
        SolverConfig(
            "CG(Mult-Schwarz)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=MultiplicativeOneLevelSchwarz(
                    smoother=ApproxCholConfig(seed=opts.seed)
                ),
            ),
        ),
    ]


def _run_problems(
    problems: list[ProblemSpec],
    configs: list[SolverConfig],
) -> list[BenchmarkResult]:
    all_results: list[BenchmarkResult] = []
    for prob in problems:
        gen = get_generator(prob.generator)
        cats, n_levels, y = gen(**prob.params, seed=prob.seed)
        n_fe = len(n_levels)
        print(
            f"\nProblem: {prob.name}  ({n_fe}-FE, DOFs={sum(n_levels)}, Rows={len(cats[0])})"
        )

        for cfg in configs:
            try:
                result = run_solve(cats, n_levels, y, cfg)
                result.problem = prob.name
                all_results.append(result)
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f"  WARNING: {cfg.label} failed: {e}")
    return all_results


@suite(
    "many_components",
    description="Problems with many (near-)disconnected components",
    tags=("components", "2fe", "3fe", "4fe"),
)
def run_many_components(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            # Truly disconnected (bridge_obs=0)
            ProblemSpec(
                "discon 3fe 10c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 100,
                    "bridge_obs": 0,
                },
                opts.seed,
            ),
            # Weakly connected
            ProblemSpec(
                "weak 3fe 10c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 100,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
            # AKM near-disconnected
            ProblemSpec(
                "akm discon 5c",
                "akm_disconnected",
                {
                    "n_workers": 2000,
                    "n_firms": 100,
                    "n_years": 5,
                    "n_clusters": 5,
                    "within_mobility": 0.10,
                    "cross_cluster_rate": 0.005,
                    "n_fe": 3,
                },
                opts.seed,
            ),
        ]
    else:
        problems = [
            # --- Truly disconnected (bridge_obs=0) ---
            ProblemSpec(
                "discon 3fe 5c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 5,
                    "obs_per_cluster": 200,
                    "bridge_obs": 0,
                },
                opts.seed,
            ),
            ProblemSpec(
                "discon 3fe 10c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 0,
                },
                opts.seed,
            ),
            ProblemSpec(
                "discon 3fe 20c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 200,
                    "n_clusters": 20,
                    "obs_per_cluster": 200,
                    "bridge_obs": 0,
                },
                opts.seed,
            ),
            ProblemSpec(
                "discon 4fe 10c",
                "disconnected_kfe",
                {
                    "k": 4,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 0,
                },
                opts.seed,
            ),
            # --- Weakly connected (thin bridges) ---
            ProblemSpec(
                "weak 3fe 10c b=1",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 1,
                },
                opts.seed,
            ),
            ProblemSpec(
                "weak 3fe 10c b=3",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "weak 3fe 20c b=2",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 200,
                    "n_clusters": 20,
                    "obs_per_cluster": 200,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
            ProblemSpec(
                "weak 4fe 10c b=2",
                "disconnected_kfe",
                {
                    "k": 4,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
            # --- Clustered 3fe with near-isolation ---
            ProblemSpec(
                "clustered 3fe 10c",
                "clustered_3fe",
                {
                    "n_levels": (100, 100, 100),
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
            ProblemSpec(
                "clustered 3fe 20c",
                "clustered_3fe",
                {
                    "n_levels": (200, 200, 200),
                    "n_clusters": 20,
                    "obs_per_cluster": 200,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
            # --- AKM near-disconnected ---
            ProblemSpec(
                "akm discon 5c",
                "akm_disconnected",
                {
                    "n_workers": 5000,
                    "n_firms": 250,
                    "n_years": 10,
                    "n_clusters": 5,
                    "within_mobility": 0.10,
                    "cross_cluster_rate": 0.005,
                    "n_fe": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "akm discon 10c",
                "akm_disconnected",
                {
                    "n_workers": 5000,
                    "n_firms": 500,
                    "n_years": 10,
                    "n_clusters": 10,
                    "within_mobility": 0.10,
                    "cross_cluster_rate": 0.005,
                    "n_fe": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "akm low-mob 10c",
                "akm_disconnected",
                {
                    "n_workers": 5000,
                    "n_firms": 500,
                    "n_years": 10,
                    "n_clusters": 10,
                    "within_mobility": 0.05,
                    "cross_cluster_rate": 0.002,
                    "n_fe": 3,
                },
                opts.seed,
            ),
        ]

    configs = _make_configs(opts)
    results = _run_problems(problems, configs)
    print_table(results)
    print("\n")
    print_pivot(results)
    return results


@suite(
    "component_scaling",
    description="Scaling as number of connected components grows",
    tags=("components", "scaling"),
)
def run_component_scaling(opts: SuiteOptions) -> list[BenchmarkResult]:
    """Fix total size, vary number of clusters from 2 to 50."""
    if opts.quick:
        problems = [
            ProblemSpec(
                "3fe 2c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 2,
                    "obs_per_cluster": 500,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
            ProblemSpec(
                "3fe 10c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 100,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
            ProblemSpec(
                "3fe 20c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 20,
                    "obs_per_cluster": 50,
                    "bridge_obs": 2,
                },
                opts.seed,
            ),
        ]
    else:
        problems = [
            ProblemSpec(
                "3fe 2c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 2,
                    "obs_per_cluster": 1000,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "3fe 5c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 5,
                    "obs_per_cluster": 400,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "3fe 10c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 100,
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "3fe 20c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 200,
                    "n_clusters": 20,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "3fe 50c",
                "disconnected_kfe",
                {
                    "k": 3,
                    "n_levels": 500,
                    "n_clusters": 50,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
        ]

    configs = _make_configs(opts)
    results = _run_problems(problems, configs)
    print_table(results)
    print("\n")
    print_pivot(results)
    return results
