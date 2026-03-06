"""2-FE Laplacian benchmark suite.

Compares LSMR(diag) vs CG(Schwarz) on 2-FE problems.
"""

from __future__ import annotations

from within import (
    ApproxCholConfig,
    CG,
    GMRES,
    LSMR,
    AdditiveSchwarz,
    MultiplicativeSchwarz,
)
from .._problems import get_generator
from .._registry import SuiteOptions, suite
from .._solvers import run_solve
from .._table import print_table
from .._types import BenchmarkResult, ProblemSpec, SolverConfig


@suite(
    "laplacian_2fe",
    description="LSMR(diag) vs CG(Schwarz) on 2-FE Laplacian problems",
    tags=("2fe", "laplacian"),
)
def run_laplacian_2fe(opts: SuiteOptions) -> list[BenchmarkResult]:
    if opts.quick:
        problems = [
            ProblemSpec("Chain n=50", "chain_2fe", {"n_levels": 50}, opts.seed),
            ProblemSpec("Chain n=100", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec(
                "Expander n=100 d=3",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
        ]
    else:
        problems = [
            ProblemSpec("Chain n=50", "chain_2fe", {"n_levels": 50}, opts.seed),
            ProblemSpec("Chain n=100", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("Chain n=200", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec("Chain n=500", "chain_2fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("Chain n=1000", "chain_2fe", {"n_levels": 1000}, opts.seed),
            ProblemSpec("Chain n=2000", "chain_2fe", {"n_levels": 2000}, opts.seed),
            ProblemSpec("Star n=100", "star_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("Star n=200", "star_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec(
                "Expander n=100 d=3",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Expander n=200 d=3",
                "expander_2fe",
                {"n_levels": 200, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Expander n=500 d=3",
                "expander_2fe",
                {"n_levels": 500, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Expander n=1000 d=3",
                "expander_2fe",
                {"n_levels": 1000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Expander n=2000 d=3",
                "expander_2fe",
                {"n_levels": 2000, "degree": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Chain n=50 (dense)",
                "chain_2fe",
                {"n_levels": 50, "obs_per_edge": 10},
                opts.seed,
            ),
            ProblemSpec(
                "Expander n=100 d=10",
                "expander_2fe",
                {"n_levels": 100, "degree": 10},
                opts.seed,
            ),
        ]

    configs = [
        SolverConfig("LSMR(diag)", LSMR(tol=opts.tol, maxiter=opts.maxiter)),
        SolverConfig(
            "CG(Schwarz)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=AdditiveSchwarz(
                    smoother=ApproxCholConfig(seed=opts.seed)
                ),
            ),
        ),
        SolverConfig(
            "GMRES(Mult-Schwarz)",
            GMRES(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=MultiplicativeSchwarz(
                    smoother=ApproxCholConfig(seed=opts.seed)
                ),
            ),
        ),
        SolverConfig(
            "CG(Mult-Schwarz)",
            CG(
                tol=opts.tol,
                maxiter=opts.maxiter,
                preconditioner=MultiplicativeSchwarz(
                    smoother=ApproxCholConfig(seed=opts.seed)
                ),
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
            all_results.append(result)

    print_table(all_results)

    # Summary ratio table
    print(f"\n{'Problem':<30} {'LSMR':>7} {'CG':>7} {'Ratio':>7}")
    print("-" * 55)
    for prob in problems:
        d = next(
            (
                r
                for r in all_results
                if r.problem == prob.name and r.config == "LSMR(diag)"
            ),
            None,
        )
        a = next(
            (
                r
                for r in all_results
                if r.problem == prob.name and r.config == "CG(Schwarz)"
            ),
            None,
        )
        if d and a and a.iterations > 0:
            ratio = d.iterations / a.iterations
            d_str = f"{d.iterations}{'*' if not d.converged else ''}"
            a_str = f"{a.iterations}{'*' if not a.converged else ''}"
            print(f"{prob.name:<30} {d_str:>7} {a_str:>7} {ratio:>7.1f}x")

    return all_results
