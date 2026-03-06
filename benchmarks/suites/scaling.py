"""Scaling benchmark suites."""

from __future__ import annotations

import numpy as np

from .._registry import SuiteOptions, suite
from .._solver_presets import standard_solver_configs
from .._solvers import run_solve
from .._table import print_pivot, print_table
from .._types import BenchmarkResult


def _run_scaling_problems(
    configs_list: list[tuple[str, list[int], int]],
    opts: SuiteOptions,
) -> list[BenchmarkResult]:
    """Run a list of (name, n_levels, n_rows) configs through the standard solve paths."""
    solver_configs = standard_solver_configs(opts)

    all_results: list[BenchmarkResult] = []
    for name, n_levels, n_rows in configs_list:
        rng = np.random.default_rng(opts.seed)
        cats = [rng.integers(0, nl, size=n_rows) for nl in n_levels]
        x_true = rng.standard_normal(sum(n_levels))
        y = np.zeros(n_rows)
        offset = 0
        for f, nl in enumerate(n_levels):
            y += x_true[offset + cats[f]]
            offset += nl
        print(f"\n  {name}: DOFs={sum(n_levels)}, Rows={n_rows}")

        for cfg in solver_configs:
            try:
                result = run_solve(cats, n_levels, y, cfg)
                result.problem = name
                all_results.append(result)
            except Exception as e:
                print(f"    WARNING: {cfg.label} failed: {e}")

    return all_results


@suite(
    "scaling",
    description="Scaling benchmark across 2-FE and 3-FE problem sizes",
    tags=("2fe", "3fe", "scaling"),
)
def run_scaling(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        configs = [
            ("2f quick", [1000, 1000], 100_000),
            ("3f quick", [800, 800, 400], 100_000),
        ]
    else:
        configs = [
            # 2-FE
            ("2f balanced 5K", [2500, 2500], 200_000),
            ("2f asymmetric 5K", [4000, 1000], 200_000),
            ("2f balanced 20K", [10000, 10000], 500_000),
            # 3-FE
            ("3f balanced 5K", [2000, 2000, 1000], 200_000),
            ("3f pyramid 5K", [3000, 1500, 500], 200_000),
            ("3f balanced 20K", [8000, 8000, 4000], 500_000),
        ]

    results = _run_scaling_problems(configs, opts)
    print_table(results)
    print("\n")
    print_pivot(results)
    return results


@suite(
    "scaling_2fe",
    description="Specialized 2-FE scaling (chain, star, barbell, expander)",
    tags=("2fe", "scaling"),
)
def run_scaling_2fe(opts: SuiteOptions) -> list[BenchmarkResult]:
    from .._problems import get_generator

    if opts.quick:
        problems = [
            ("chain 100", "chain_2fe", {"n_levels": 100}),
            ("expander 100 d=3", "expander_2fe", {"n_levels": 100, "degree": 3}),
        ]
    else:
        problems = [
            ("chain 100", "chain_2fe", {"n_levels": 100}),
            ("chain 500", "chain_2fe", {"n_levels": 500}),
            ("chain 1000", "chain_2fe", {"n_levels": 1000}),
            ("star 200", "star_2fe", {"n_levels": 200}),
            ("barbell 100", "barbell_2fe", {"n_levels": 100}),
            ("barbell 500", "barbell_2fe", {"n_levels": 500}),
            ("barbell 1000", "barbell_2fe", {"n_levels": 1000}),
            ("expander 200 d=3", "expander_2fe", {"n_levels": 200, "degree": 3}),
            ("expander 500 d=3", "expander_2fe", {"n_levels": 500, "degree": 3}),
            ("expander 1000 d=3", "expander_2fe", {"n_levels": 1000, "degree": 3}),
            ("grid 20x20", "grid_2fe", {"n_side": 20}),
        ]

    solver_configs = standard_solver_configs(opts)

    all_results: list[BenchmarkResult] = []
    for name, gen_key, params in problems:
        gen = get_generator(gen_key)
        cats, n_levels, y = gen(**params, seed=opts.seed)
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
