"""Preconditioner correctness verification suite."""

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
from .._table import print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig, run_solve

RESIDUAL_THRESHOLD = 1e-6


def _problems(quick: bool, seed: int) -> list[ProblemSpec]:
    if quick:
        return [
            ProblemSpec("chain-50 2fe", "chain_2fe", {"n_levels": 50}, seed),
            ProblemSpec("barbell-50 2fe", "barbell_2fe", {"n_levels": 50}, seed),
            ProblemSpec(
                "sparse-3e-50^3",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                seed,
            ),
            ProblemSpec("chain-3fe-50", "chain_3fe", {"n_levels": 50}, seed),
        ]
    return [
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
            "barbell-3fe-100", "barbell_3fe", {"n_levels": 100, "bridge_width": 4}, seed
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
    ]


@suite(
    "verify",
    description="Verify preconditioner correctness on 2-FE and 3-FE problems",
    tags=("2fe", "3fe", "correctness"),
)
def run_verify(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = _problems(opts.quick, opts.seed)

    schur = SchurComplement(
        approx_chol=ApproxCholConfig(seed=opts.seed),
        approx_schur=ApproxSchurConfig(seed=opts.seed),
    )
    configs = [
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

    all_results: list[BenchmarkResult] = []
    for prob in problems:
        gen = get_generator(prob.generator)
        cats, n_levels, y = gen(**prob.params, seed=prob.seed)
        print(f"\nProblem: {prob.name}  (DOFs={sum(n_levels)}, Rows={len(cats[0])})")

        for cfg in configs:
            result = run_solve(cats, n_levels, y, cfg)
            result.problem = prob.name
            result.passed = (
                result.converged and result.final_residual < RESIDUAL_THRESHOLD
            )
            all_results.append(result)

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
