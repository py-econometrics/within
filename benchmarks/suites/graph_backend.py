"""ApproxChol variant comparison suite.

Compares AC vs AC2 smoothing across topologies.
"""

from __future__ import annotations

from within import (
    AdditiveSchwarz,
    ApproxCholConfig,
    ApproxSchurConfig,
    CG,
    SchurComplement,
)
from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solvers import run_solve
from .._table import print_pivot, print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig


@suite(
    "graph_backend_comparison",
    description="ApproxChol variant comparison across topologies",
    tags=("2fe", "3fe", "ac"),
)
def run_graph_backend_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec("chain 100 2fe", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("chain 500 2fe", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("star 100 2fe", "star_2fe", {"n_levels": 100}, opts.seed),
        ]
    else:
        problems = [
            ProblemSpec("chain 500 2fe", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("chain 2000 2fe", "chain_2fe", {"n_levels": 2000}, opts.seed),
            ProblemSpec("chain 5000 2fe", "chain_2fe", {"n_levels": 5000}, opts.seed),
            ProblemSpec("chain 10000 2fe", "chain_2fe", {"n_levels": 10000}, opts.seed),
            ProblemSpec("star 500 2fe", "star_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("star 2000 2fe", "star_2fe", {"n_levels": 2000}, opts.seed),
            ProblemSpec("star 5000 2fe", "star_2fe", {"n_levels": 5000}, opts.seed),
            ProblemSpec(
                "expander 500 d=3",
                "expander_2fe",
                {"n_levels": 500, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 2000 d=3",
                "expander_2fe",
                {"n_levels": 2000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 5000 d=3",
                "expander_2fe",
                {"n_levels": 5000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "expander 500 d=10",
                "expander_2fe",
                {"n_levels": 500, "degree": 10},
                opts.seed,
            ),
            ProblemSpec(
                "expander 2000 d=10",
                "expander_2fe",
                {"n_levels": 2000, "degree": 10},
                opts.seed,
            ),
            ProblemSpec("barbell 500 2fe", "barbell_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec(
                "barbell 2000 2fe", "barbell_2fe", {"n_levels": 2000}, opts.seed
            ),
            ProblemSpec("grid 50x50 2fe", "grid_2fe", {"n_side": 50}, opts.seed),
            ProblemSpec("grid 100x100 2fe", "grid_2fe", {"n_side": 100}, opts.seed),
            ProblemSpec(
                "sparse 200^3 3fe",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "sparse 500^3 3fe",
                "sparse_3fe",
                {"n_levels": (500, 500, 500), "edges_per_level": 3},
                opts.seed,
            ),
        ]

    def _schur(split: int) -> SchurComplement:
        return SchurComplement(
            approx_chol=ApproxCholConfig(seed=opts.seed, split=split),
            approx_schur=ApproxSchurConfig(seed=opts.seed),
        )

    configs = [
        SolverConfig(
            "ac",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=_schur(1)),
            ),
        ),
        SolverConfig(
            "ac2",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(local_solver=_schur(2)),
            ),
        ),
    ]

    if opts.quick:
        configs = [c for c in configs if c.label in {"ac", "ac2"}]

    all_results: list[BenchmarkResult] = []
    for prob in problems:
        gen = get_generator(prob.generator)
        cats, n_levels, y = gen(**prob.params, seed=prob.seed)
        print(f"\nProblem: {prob.name}  (DOFs={sum(n_levels)}, Rows={len(cats[0])})")

        for cfg in configs:
            try:
                result = run_solve(cats, n_levels, y, cfg)
                result.problem = prob.name
                all_results.append(result)
            except Exception as e:
                print(f"  WARNING: {cfg.label} failed: {e}")

    print_table(all_results)
    print("\nSetup time pivot:")
    print_pivot(all_results, value="setup_time")
    print("\nIteration count pivot:")
    print_pivot(all_results, value="iterations")
    return all_results
