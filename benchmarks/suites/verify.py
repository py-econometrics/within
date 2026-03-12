"""Preconditioner correctness verification suite."""

from __future__ import annotations

from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    MultiplicativeSchwarz,
    SchurComplement,
)
from .._framework import (
    BenchmarkResult,
    ProblemSpec,
    SolverConfig,
    SuiteOptions,
    benchmark_cg,
    benchmark_gmres,
    make_additive_schwarz,
    run_problem_set,
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
            # 2-FE
            ProblemSpec("chain-100 2fe", "chain_2fe", {"n_levels": 100}, seed),
            ProblemSpec("chain-250 2fe", "chain_2fe", {"n_levels": 250}, seed),
            ProblemSpec("chain-500 2fe", "chain_2fe", {"n_levels": 500}, seed),
            ProblemSpec("barbell-100 2fe", "barbell_2fe", {"n_levels": 100}, seed),
            ProblemSpec("barbell-250 2fe", "barbell_2fe", {"n_levels": 250}, seed),
            ProblemSpec("barbell-500 2fe", "barbell_2fe", {"n_levels": 500}, seed),
            # 3-FE
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
    description="Verify preconditioner correctness on 2-FE and 3-FE problems",
    tags=("2fe", "3fe", "correctness"),
)
def run_verify(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = _problems(opts)

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    configs = [
        SolverConfig(
            "CG(Schwarz)",
            benchmark_cg(opts),
            preconditioner=make_additive_schwarz(local_solver=schur),
        ),
        SolverConfig(
            "GMRES(Mult-Schwarz)",
            benchmark_gmres(opts),
            preconditioner=MultiplicativeSchwarz(local_solver=schur),
        ),
    ]

    all_results = run_problem_set(problems, configs, opts)
    for r in all_results:
        r.passed = r.converged and r.final_residual < RESIDUAL_THRESHOLD

    print_table(
        all_results,
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
