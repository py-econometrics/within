"""Benchmarks for mixed subdomain-size and heavy-tail FE regimes.

These cases are aimed at additive scheduling and backend tuning. They sit
between the existing "many small" and "few large" extremes by creating one or
two dominant local regions together with a long tail of smaller ones.
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
    "subdomain_regimes",
    description="Dominant hub and dense-block FE regimes with long-tail structure",
    tags=("subdomains", "imbalance", "3fe", "4fe"),
)
def run_subdomain_regimes(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = opts.select(
        smoke=[
            ProblemSpec(
                "hub 3fe 120x120x60",
                "hub_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [120, 120, 60],
                    "n_rows": 6000,
                    "hub_factor": 0,
                    "hub_share": 0.85,
                },
                opts.seed,
            ),
            ProblemSpec(
                "block 3fe 120x96x72",
                "dominant_block_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [120, 96, 72],
                    "n_rows": 6000,
                    "block_levels": [6, 6, 4],
                    "block_share": 0.80,
                },
                opts.seed,
            ),
            ProblemSpec(
                "hub 4fe 60^4",
                "hub_kfe",
                {
                    "k": 4,
                    "n_levels_per_factor": [60, 60, 60, 60],
                    "n_rows": 7000,
                    "hub_factor": 1,
                    "hub_share": 0.82,
                },
                opts.seed,
            ),
        ],
        iterate=[
            ProblemSpec(
                "hub 3fe 300x240x120",
                "hub_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [300, 240, 120],
                    "n_rows": 30000,
                    "hub_factor": 0,
                    "hub_share": 0.90,
                },
                opts.seed,
            ),
            ProblemSpec(
                "hub 3fe tail-heavy",
                "hub_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [400, 300, 80],
                    "n_rows": 30000,
                    "hub_factor": 2,
                    "hub_share": 0.88,
                },
                opts.seed,
            ),
            ProblemSpec(
                "block 3fe 300x240x180",
                "dominant_block_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [300, 240, 180],
                    "n_rows": 30000,
                    "block_levels": [8, 8, 6],
                    "block_share": 0.82,
                },
                opts.seed,
            ),
            ProblemSpec(
                "block 4fe 120^4",
                "dominant_block_kfe",
                {
                    "k": 4,
                    "n_levels_per_factor": [120, 120, 120, 120],
                    "n_rows": 35000,
                    "block_levels": [5, 5, 5, 5],
                    "block_share": 0.78,
                },
                opts.seed,
            ),
            ProblemSpec(
                "hub 4fe asym",
                "hub_kfe",
                {
                    "k": 4,
                    "n_levels_per_factor": [240, 120, 80, 40],
                    "n_rows": 30000,
                    "hub_factor": 0,
                    "hub_share": 0.90,
                },
                opts.seed,
            ),
        ],
        full=[
            ProblemSpec(
                "hub 3fe 300x240x120",
                "hub_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [300, 240, 120],
                    "n_rows": 30000,
                    "hub_factor": 0,
                    "hub_share": 0.90,
                },
                opts.seed,
            ),
            ProblemSpec(
                "hub 3fe tail-heavy",
                "hub_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [400, 300, 80],
                    "n_rows": 30000,
                    "hub_factor": 2,
                    "hub_share": 0.88,
                },
                opts.seed,
            ),
            ProblemSpec(
                "block 3fe 300x240x180",
                "dominant_block_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [300, 240, 180],
                    "n_rows": 30000,
                    "block_levels": [8, 8, 6],
                    "block_share": 0.82,
                },
                opts.seed,
            ),
            ProblemSpec(
                "block 3fe 500x400x250",
                "dominant_block_kfe",
                {
                    "k": 3,
                    "n_levels_per_factor": [500, 400, 250],
                    "n_rows": 60000,
                    "block_levels": [10, 10, 8],
                    "block_share": 0.84,
                },
                opts.seed,
            ),
            ProblemSpec(
                "block 4fe 120^4",
                "dominant_block_kfe",
                {
                    "k": 4,
                    "n_levels_per_factor": [120, 120, 120, 120],
                    "n_rows": 35000,
                    "block_levels": [5, 5, 5, 5],
                    "block_share": 0.78,
                },
                opts.seed,
            ),
            ProblemSpec(
                "hub 4fe asym",
                "hub_kfe",
                {
                    "k": 4,
                    "n_levels_per_factor": [240, 120, 80, 40],
                    "n_rows": 30000,
                    "hub_factor": 0,
                    "hub_share": 0.90,
                },
                opts.seed,
            ),
        ],
    )

    results = run_problem_set(problems, standard_solver_configs(opts), opts)
    print_table(results)
    print("\n")
    print_pivot(results)
    print()
    print_pivot(results, value="solve_time")
    return results
