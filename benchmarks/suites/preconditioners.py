"""Preconditioner comparison suites.

Three suites:
- ``preconditioners_3fe`` — CG vs GMRES Schwarz on 3-FE problems (includes
  one-level baseline problems)
- ``preconditioner_comparison`` — unpreconditioned vs preconditioned on 2-FE
  and 3-FE (includes iteration-reduction problems with summary)
"""

from __future__ import annotations

from .._framework import (
    BenchmarkResult,
    ProblemSpec,
    SolverConfig,
    SuiteOptions,
    benchmark_cg,
    run_problem_set,
    standard_solver_configs,
    suite,
)
from .._table import print_pivot, print_table


@suite(
    "preconditioners_3fe",
    description="CG(additive Schwarz) vs GMRES(multiplicative Schwarz) on 3-FE problems",
    tags=("3fe", "precond", "baseline"),
)
def run_preconditioners_3fe(opts: SuiteOptions) -> list[BenchmarkResult]:
    problems = opts.select(
        smoke=[
            ProblemSpec(
                "Sparse (50^3, 3e)",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Imbalanced (50^3)",
                "imbalanced_3fe",
                {"n_levels": (50, 50, 50), "n_rows": 5000},
                opts.seed,
            ),
            # From one_level_baseline
            ProblemSpec(
                "Clustered (100^3)",
                "clustered_3fe",
                {
                    "n_levels": (100, 100, 100),
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec("Barbell 3fe 250", "barbell_3fe", {"n_levels": 250}, opts.seed),
            ProblemSpec("Chain 3fe 250", "chain_3fe", {"n_levels": 250}, opts.seed),
        ],
        iterate=[
            ProblemSpec(
                "Sparse (100^3, 3e)",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "V.Sparse (200^3, 2e)",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 2},
                opts.seed,
            ),
            ProblemSpec(
                "Clustered (200^3)",
                "clustered_3fe",
                {
                    "n_levels": (200, 200, 200),
                    "n_clusters": 10,
                    "obs_per_cluster": 500,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec("Barbell 3fe 500", "barbell_3fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("Chain 3fe 500", "chain_3fe", {"n_levels": 500}, opts.seed),
            ProblemSpec(
                "Imbalanced (200^3)",
                "imbalanced_3fe",
                {"n_levels": (200, 200, 200), "n_rows": 30000},
                opts.seed,
            ),
            ProblemSpec(
                "AKM disconnected",
                "akm_disconnected",
                {"n_workers": 5000, "n_firms": 300, "n_years": 10, "n_clusters": 5},
                opts.seed,
            ),
        ],
        full=[
            ProblemSpec(
                "Sparse (50^3, 3e)",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse (50^3, 5e)",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 5},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse (100^3, 3e)",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Sparse (100^3, 5e)",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 5},
                opts.seed,
            ),
            ProblemSpec(
                "V.Sparse (200^3, 2e)",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 2},
                opts.seed,
            ),
            ProblemSpec(
                "V.Sparse (200^3, 3e)",
                "sparse_3fe",
                {"n_levels": (200, 200, 200), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec(
                "Clustered (100^3)",
                "clustered_3fe",
                {
                    "n_levels": (100, 100, 100),
                    "n_clusters": 10,
                    "obs_per_cluster": 200,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec(
                "Imbalanced (100^3)",
                "imbalanced_3fe",
                {"n_levels": (100, 100, 100), "n_rows": 10000},
                opts.seed,
            ),
            # From one_level_baseline
            ProblemSpec(
                "Clustered (200^3)",
                "clustered_3fe",
                {
                    "n_levels": (200, 200, 200),
                    "n_clusters": 10,
                    "obs_per_cluster": 500,
                    "bridge_obs": 3,
                },
                opts.seed,
            ),
            ProblemSpec("Barbell 3fe 250", "barbell_3fe", {"n_levels": 250}, opts.seed),
            ProblemSpec("Barbell 3fe 500", "barbell_3fe", {"n_levels": 500}, opts.seed),
            ProblemSpec("Chain 3fe 250", "chain_3fe", {"n_levels": 250}, opts.seed),
            ProblemSpec("Chain 3fe 500", "chain_3fe", {"n_levels": 500}, opts.seed),
            ProblemSpec(
                "Imbalanced (200^3)",
                "imbalanced_3fe",
                {"n_levels": (200, 200, 200), "n_rows": 30000},
                opts.seed,
            ),
            ProblemSpec(
                "AKM power-law",
                "akm_power_law",
                {"n_workers": 5000, "n_firms": 300, "n_years": 10},
                opts.seed,
            ),
            ProblemSpec(
                "AKM disconnected",
                "akm_disconnected",
                {"n_workers": 5000, "n_firms": 300, "n_years": 10, "n_clusters": 5},
                opts.seed,
            ),
        ],
    )

    all_results = run_problem_set(problems, standard_solver_configs(opts), opts)
    print_table(all_results)
    print("\n")
    print_pivot(all_results)
    print()
    print_pivot(all_results, value="setup_time")
    print()
    print_pivot(all_results, value="solve_time")
    return all_results


@suite(
    "preconditioner_comparison",
    description="Unpreconditioned CG vs Schwarz-preconditioned CG/GMRES with iteration reduction",
    tags=("2fe", "3fe", "precond"),
)
def run_preconditioner_comparison(opts: SuiteOptions) -> list[BenchmarkResult]:
    maxiter = min(opts.maxiter, 1000)

    problems = opts.select(
        smoke=[
            ProblemSpec("chain 50 2fe", "chain_2fe", {"n_levels": 50}, opts.seed),
            ProblemSpec(
                "sparse 50^3 3fe",
                "sparse_3fe",
                {"n_levels": (50, 50, 50), "edges_per_level": 3},
                opts.seed,
            ),
            # From iteration_reduction
            ProblemSpec(
                "2-FE 30x40",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40], "n_rows": 2000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 30x40x25",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40, 25], "n_rows": 2000},
                opts.seed,
            ),
        ],
        iterate=[
            ProblemSpec("chain 200 2fe", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec(
                "expander 100 2fe",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
            ProblemSpec("barbell 100 2fe", "barbell_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec(
                "sparse 100^3 3fe",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec("chain 100 3fe", "chain_3fe", {"n_levels": 100}, opts.seed),
            ProblemSpec(
                "2-FE 50x80",
                "uniform_kfe",
                {"n_levels_per_factor": [50, 80], "n_rows": 5000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 50x80x30",
                "uniform_kfe",
                {"n_levels_per_factor": [50, 80, 30], "n_rows": 5000},
                opts.seed,
            ),
        ],
        full=[
            ProblemSpec("chain 100 2fe", "chain_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec("chain 200 2fe", "chain_2fe", {"n_levels": 200}, opts.seed),
            ProblemSpec(
                "expander 100 2fe",
                "expander_2fe",
                {"n_levels": 100, "degree": 3},
                opts.seed,
            ),
            ProblemSpec("barbell 100 2fe", "barbell_2fe", {"n_levels": 100}, opts.seed),
            ProblemSpec(
                "sparse 100^3 3fe",
                "sparse_3fe",
                {"n_levels": (100, 100, 100), "edges_per_level": 3},
                opts.seed,
            ),
            ProblemSpec("chain 100 3fe", "chain_3fe", {"n_levels": 100}, opts.seed),
            # From iteration_reduction
            ProblemSpec(
                "2-FE 30x40",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40], "n_rows": 2000},
                opts.seed,
            ),
            ProblemSpec(
                "2-FE 50x80",
                "uniform_kfe",
                {"n_levels_per_factor": [50, 80], "n_rows": 5000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 30x40x25",
                "uniform_kfe",
                {"n_levels_per_factor": [30, 40, 25], "n_rows": 2000},
                opts.seed,
            ),
            ProblemSpec(
                "3-FE 50x80x30",
                "uniform_kfe",
                {"n_levels_per_factor": [50, 80, 30], "n_rows": 5000},
                opts.seed,
            ),
        ],
    )

    configs = [
        SolverConfig(
            "CG(none)",
            benchmark_cg(opts, maxiter=maxiter),
        ),
        *standard_solver_configs(opts),
    ]

    all_results = run_problem_set(problems, configs, opts)
    print_table(all_results)
    print("\n")
    print_pivot(all_results)

    # Iteration-reduction summary
    print(
        f"\n{'Problem':<18} | {'Solver':<10} | {'Unprec':>7} | {'Prec':>7} | {'Reduction':>9}"
    )
    print("-" * 65)
    for prob in problems:
        label_none = "CG(none)"
        label_prec = "CG(Schwarz)"
        r_none = next(
            (
                r
                for r in all_results
                if r.problem == prob.name and r.config == label_none
            ),
            None,
        )
        r_prec = next(
            (
                r
                for r in all_results
                if r.problem == prob.name and r.config == label_prec
            ),
            None,
        )
        if r_none and r_prec and r_none.iterations > 0:
            reduction = f"{100 * (1 - r_prec.iterations / r_none.iterations):6.1f}%"
        else:
            reduction = "N/A"
        i_none = r_none.iterations if r_none else 0
        i_prec = r_prec.iterations if r_prec else 0
        print(
            f"{prob.name:<18} | {'CG':<10} | {i_none:>7} | {i_prec:>7} | {reduction:>9}"
        )

    return all_results
