"""Scaling benchmark suites."""

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
    "scaling",
    description="Scaling benchmark across 2-FE and 3-FE problem sizes",
    tags=("2fe", "3fe", "scaling"),
)
def run_scaling(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = opts.select(
        smoke=[
            ProblemSpec(
                "2f quick",
                "uniform_kfe",
                {"n_levels_per_factor": [1000, 1000], "n_rows": 100_000},
                opts.seed,
            ),
            ProblemSpec(
                "3f quick",
                "uniform_kfe",
                {"n_levels_per_factor": [800, 800, 400], "n_rows": 100_000},
                opts.seed,
            ),
        ],
        iterate=[
            ProblemSpec(
                "2f balanced 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [2500, 2500], "n_rows": 200_000},
                opts.seed,
            ),
            ProblemSpec(
                "2f asymmetric 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [4000, 1000], "n_rows": 200_000},
                opts.seed,
            ),
            ProblemSpec(
                "3f balanced 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [2000, 2000, 1000], "n_rows": 200_000},
                opts.seed,
            ),
            ProblemSpec(
                "3f pyramid 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [3000, 1500, 500], "n_rows": 200_000},
                opts.seed,
            ),
        ],
        full=[
            # 2-FE
            ProblemSpec(
                "2f balanced 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [2500, 2500], "n_rows": 200_000},
                opts.seed,
            ),
            ProblemSpec(
                "2f asymmetric 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [4000, 1000], "n_rows": 200_000},
                opts.seed,
            ),
            ProblemSpec(
                "2f balanced 20K",
                "uniform_kfe",
                {"n_levels_per_factor": [10000, 10000], "n_rows": 500_000},
                opts.seed,
            ),
            # 3-FE
            ProblemSpec(
                "3f balanced 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [2000, 2000, 1000], "n_rows": 200_000},
                opts.seed,
            ),
            ProblemSpec(
                "3f pyramid 5K",
                "uniform_kfe",
                {"n_levels_per_factor": [3000, 1500, 500], "n_rows": 200_000},
                opts.seed,
            ),
            ProblemSpec(
                "3f balanced 20K",
                "uniform_kfe",
                {"n_levels_per_factor": [8000, 8000, 4000], "n_rows": 500_000},
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
    "scaling_2fe",
    description="2-FE scaling across topologies (chain, star, barbell, expander, grid)",
    tags=("2fe", "scaling", "laplacian"),
)
def run_scaling_2fe(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = opts.select(
        smoke=[
            ProblemSpec("chain 50", "chain_2fe", {"n_levels": 50}, opts.seed),
            ProblemSpec("chain 100", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec(
                "expander 100 d=3",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
        ],
        iterate=[
            ProblemSpec("chain 200", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec("chain 1000", "chain_2fe", {"n_levels": 1000}, opts.seed),
            ProblemSpec("star 200", "star_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec("barbell 500", "barbell_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec(
                "expander 500 d=3",
                "expander_2fe",
                {"n_levels": 500, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 1000 d=3",
                "expander_2fe",
                {"n_levels": 1000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec("grid 20x20", "grid_2fe", {"n_side": 20}, opts.seed),
        ],
        full=[
            ProblemSpec("chain 50", "chain_2fe", {"n_levels": 50}, opts.seed),
            ProblemSpec("chain 100", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("chain 200", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec("chain 500", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("chain 1000", "chain_2fe", {"n_levels": 1000}, opts.seed),
            ProblemSpec("chain 2000", "chain_2fe", {"n_levels": 2000}, opts.seed),
            ProblemSpec(
                "chain 50 (dense)",
                "chain_2fe",
                {"n_levels": 50, "obs_per_edge": 10},
                opts.seed,
            ),
            ProblemSpec("star 100", "star_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("star 200", "star_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec("barbell 100", "barbell_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("barbell 500", "barbell_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("barbell 1000", "barbell_2fe", {"n_levels": 1000}, opts.seed),
            ProblemSpec(
                "expander 100 d=3",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 200 d=3",
                "expander_2fe",
                {"n_levels": 200, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 500 d=3",
                "expander_2fe",
                {"n_levels": 500, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 1000 d=3",
                "expander_2fe",
                {"n_levels": 1000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 2000 d=3",
                "expander_2fe",
                {"n_levels": 2000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 100 d=10",
                "expander_2fe",
                {"n_levels": 100, "degree": 10},
                opts.seed,
            ),
            ProblemSpec("grid 20x20", "grid_2fe", {"n_side": 20}, opts.seed),
        ],
    )

    configs = standard_solver_configs(opts)
    results = run_problem_set(problems, configs, opts)
    print_table(results)
    print("\n")
    print_pivot(results)
    return results
