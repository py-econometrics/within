"""Observation store layout comparison suite.

Compares factor_major, row_major, and compressed backends across problem types
with varying duplication rates.
"""

from __future__ import annotations

from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solver_presets import standard_solver_configs
from .._solvers import run_solve
from .._table import print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig

LAYOUTS = ["factor_major", "row_major", "compressed"]
RESIDUAL_THRESHOLD = 1e-6


def _problems(quick: bool, seed: int) -> list[ProblemSpec]:
    if quick:
        return [
            ProblemSpec("chain-50 2fe", "chain_2fe", {"n_levels": 50}, seed),
            ProblemSpec("grid-10 2fe", "grid_2fe", {"n_side": 10}, seed),
            ProblemSpec(
                "sparse-3fe-30^3",
                "sparse_3fe",
                {"n_levels": (30, 30, 30), "edges_per_level": 3},
                seed,
            ),
            ProblemSpec(
                "akm-power-1K",
                "akm_power_law",
                {"n_workers": 1000, "n_firms": 50, "n_years": 5},
                seed,
            ),
        ]
    return [
        # 2-FE — low duplication baseline
        ProblemSpec("chain-100 2fe", "chain_2fe", {"n_levels": 100}, seed),
        ProblemSpec("star-100 2fe", "star_2fe", {"n_levels": 100}, seed),
        ProblemSpec("grid-20 2fe", "grid_2fe", {"n_side": 20}, seed),
        # 3-FE — moderate duplication
        ProblemSpec(
            "sparse-3fe-50^3",
            "sparse_3fe",
            {"n_levels": (50, 50, 50), "edges_per_level": 3},
            seed,
        ),
        # AKM — high duplication (compressed sweet spot)
        ProblemSpec(
            "akm-power-5K",
            "akm_power_law",
            {"n_workers": 5000, "n_firms": 200, "n_years": 10},
            seed,
        ),
        ProblemSpec(
            "akm-realistic-5K",
            "akm_realistic",
            {"n_workers": 5000, "n_firms": 200, "n_years": 10},
            seed,
        ),
        # Uniform large — stress test
        ProblemSpec(
            "uniform-500x500-50K",
            "uniform_kfe",
            {"n_levels_per_factor": [500, 500], "n_rows": 50000},
            seed,
        ),
    ]


def _configs(opts: SuiteOptions) -> list[SolverConfig]:
    return standard_solver_configs(opts)


@suite(
    "observation_store_comparison",
    description="Compare factor_major / row_major / compressed backends",
    tags=("layout", "observation_store"),
)
def run_observation_store(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = _problems(opts.quick, opts.seed)
    configs = _configs(opts)

    all_results: list[BenchmarkResult] = []
    for prob in problems:
        gen = get_generator(prob.generator)
        cats, n_levels, y = gen(**prob.params, seed=prob.seed)
        print(f"\nProblem: {prob.name}  (DOFs={sum(n_levels)}, Rows={len(cats[0])})")

        for cfg in configs:
            for layout in LAYOUTS:
                result = run_solve(cats, n_levels, y, cfg, layout=layout)
                result.problem = prob.name
                result.config = f"{cfg.label}/{layout}"
                result.passed = (
                    result.converged and result.final_residual < RESIDUAL_THRESHOLD
                )
                all_results.append(result)

    print_table(
        all_results,
        columns=[
            "problem",
            "config",
            "setup_time",
            "solve_time",
            "iterations",
            "final_residual",
            "converged",
        ],
    )

    # Correctness summary
    n_pass = sum(1 for r in all_results if r.passed)
    n_fail = sum(1 for r in all_results if not r.passed)
    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"\nCorrectness: {n_pass}/{len(all_results)} PASS, {n_fail} FAIL  [{status}]")
    if n_fail:
        for r in all_results:
            if not r.passed:
                print(
                    f"  FAIL: {r.problem} / {r.config}: residual={r.final_residual:.2e}"
                )

    return all_results
