"""Many connected-components suites.

Tests solver behaviour on problems with many (near-)disconnected
components in the bipartite factor graph.  These arise naturally in
employer-employee data with regional labour markets and low mobility.
"""

from __future__ import annotations

from .._framework import (
    BenchmarkResult,
    ProblemSpec,
    SuiteOptions,
    run_problem_set,
    standard_solver_configs,
    suite,
)
from .._table import print_pivot, print_table


@suite(
    "many_components",
    description="Problems with many (near-)disconnected components",
    tags=("components", "2fe", "3fe", "4fe"),
)
def run_many_components(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = opts.select(
        smoke=[
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
        ],
        iterate=[
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
        ],
        full=[
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
        ],
    )

    configs = standard_solver_configs(opts)
    results = run_problem_set(problems, configs, opts)
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
    problems = opts.select(
        smoke=[
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
        ],
        iterate=[
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
        ],
        full=[
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
        ],
    )

    configs = standard_solver_configs(opts)
    results = run_problem_set(problems, configs, opts)
    print_table(results)
    print("\n")
    print_pivot(results)
    return results
