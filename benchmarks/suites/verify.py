"""LSMR Schwarz correctness verification suite."""

from __future__ import annotations

from .._framework import (
    BenchmarkResult,
    ProblemSpec,
    SuiteOptions,
    run_problem_set,
    standard_solver_configs,
    suite,
)
from .._table import print_table

RESIDUAL_THRESHOLD = 1e-6


def _problems(opts: SuiteOptions) -> list[ProblemSpec]:
    seed = opts.seed
    return opts.select(
        smoke=[
            ProblemSpec("chain-50 2fe", "chain_2fe", {"n_levels": 50}, seed),
            ProblemSpec("barbell-50 2fe", "barbell_2fe", {"n_levels": 50}, seed),
            ProblemSpec(
                "sparse-3e-50^3",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                seed,
            ),
            ProblemSpec("chain-3fe-50", "chain_3fe", {"n_levels": 50}, seed),
        ],
        iterate=[
            ProblemSpec("chain-250 2fe", "chain_2fe", {"n_levels": 250}, seed),
            ProblemSpec("barbell-250 2fe", "barbell_2fe", {"n_levels": 250}, seed),
            ProblemSpec(
                "sparse-3e-100^3",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                seed,
            ),
            ProblemSpec(
                "imbal-100^3-10K",
                "imbalanced_3fe",
                {"n_levels": (100, 100, 100), "n_rows": 10000},
                seed,
            ),
            ProblemSpec("chain-3fe-250", "chain_3fe", {"n_levels": 250}, seed),
            ProblemSpec(
                "barbell-3fe-250",
                "barbell_3fe",
                {"n_levels": 250, "bridge_width": 10},
                seed,
            ),
        ],
        full=[
            ProblemSpec("chain-100 2fe", "chain_2fe", {"n_levels": 100}, seed),
            ProblemSpec("chain-250 2fe", "chain_2fe", {"n_levels": 250}, seed),
            ProblemSpec("chain-500 2fe", "chain_2fe", {"n_levels": 500}, seed),
            ProblemSpec("barbell-100 2fe", "barbell_2fe", {"n_levels": 100}, seed),
            ProblemSpec("barbell-250 2fe", "barbell_2fe", {"n_levels": 250}, seed),
            ProblemSpec("barbell-500 2fe", "barbell_2fe", {"n_levels": 500}, seed),
            ProblemSpec(
                "sparse-3e-50^3",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                seed,
            ),
            ProblemSpec(
                "sparse-3e-100^3",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                seed,
            ),
            ProblemSpec(
                "sparse-2e-50^3",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 2},
                seed,
            ),
            ProblemSpec(
                "sparse-2e-100^3",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 2},
                seed,
            ),
            ProblemSpec(
                "imbal-50^3-5K",
                "imbalanced_3fe",
                {"n_levels": (50, 50, 50), "n_rows": 5000},
                seed,
            ),
            ProblemSpec(
                "imbal-100^3-10K",
                "imbalanced_3fe",
                {"n_levels": (100, 100, 100), "n_rows": 10000},
                seed,
            ),
            ProblemSpec("chain-3fe-50", "chain_3fe", {"n_levels": 50}, seed),
            ProblemSpec("chain-3fe-100", "chain_3fe", {"n_levels": 100}, seed),
            ProblemSpec("chain-3fe-250", "chain_3fe", {"n_levels": 250}, seed),
            ProblemSpec(
                "barbell-3fe-100",
                "barbell_3fe",
                {"n_levels": 100, "bridge_width": 4},
                seed,
            ),
            ProblemSpec(
                "barbell-3fe-250",
                "barbell_3fe",
                {"n_levels": 250, "bridge_width": 10},
                seed,
            ),
            ProblemSpec(
                "barbell-3fe-500",
                "barbell_3fe",
                {"n_levels": 500, "bridge_width": 20},
                seed,
            ),
        ],
    )


@suite(
    "verify",
    description="Verify LSMR Schwarz correctness on 2-FE and 3-FE problems",
    tags=("2fe", "3fe", "correctness"),
)
def run_verify(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = _problems(opts)
    results = run_problem_set(problems, standard_solver_configs(opts), opts)
    for r in results:
        r.passed = r.converged and r.final_residual < RESIDUAL_THRESHOLD

    print_table(
        results,
        columns=[
            "config",
            "setup_time",
            "solve_time",
            "iterations",
            "final_residual",
            "converged",
            "passed",
        ],
    )

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"\nCorrectness: {n_pass}/{len(results)} PASS, {n_fail} FAIL  [{status}]")
    if n_fail:
        for r in results:
            if not r.passed:
                print(
                    f"  FAIL: {r.problem} / {r.config}: residual={r.final_residual:.2e}"
                )

    return results
