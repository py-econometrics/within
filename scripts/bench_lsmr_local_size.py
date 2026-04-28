"""LSMR `local_size` benchmark.

Compares LSMR with various windowed-reorthogonalization sizes on a few
representative panel-data problems. Prints iterations and wall time per
solve so the cost/benefit of windowed MGS is visible.

Two regimes are tested:

* ``+precond`` — additive-Schwarz preconditioner. Few iterations, so the
  reorthogonalization mostly costs without saving anything; this is the
  no-regression check.
* ``-precond`` — unpreconditioned. Many iterations on ill-conditioned
  topologies, so v-orthogonality drift bites and windowed MGS recovers
  iteration count + convergence.

Run from the repo root:
    PYTHONPATH=. pixi run -- python scripts/bench_lsmr_local_size.py
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from benchmarks._framework import make_additive_schwarz
from benchmarks._problems import get_generator
from within import LSMR, solve
from within._within import ApproxCholConfig, ApproxSchurConfig, SchurComplement


def run_one(
    cats: np.ndarray,
    y: np.ndarray,
    *,
    tol: float,
    maxiter: int,
    local_size: int | None,
    seed: int,
    use_precond: bool,
    n_warmup: int = 1,
    n_repeat: int = 3,
) -> tuple[float, int, bool, float]:
    """Time a single (config) solve, returning median solve_time, iters, conv, residual."""
    config = LSMR(tol=tol, maxiter=maxiter, local_size=local_size)
    if use_precond:
        schur = SchurComplement(
            approx_chol=ApproxCholConfig(seed=seed),
            approx_schur=ApproxSchurConfig(seed=seed),
        )
        precond: Any = make_additive_schwarz(local_solver=schur)
    else:
        precond = None

    times: list[float] = []
    iters = 0
    converged = True
    residual = 0.0
    for run_idx in range(n_warmup + n_repeat):
        t0 = time.perf_counter()
        r = solve(cats, y, config, preconditioner=precond)
        t1 = time.perf_counter()
        if run_idx >= n_warmup:
            times.append(t1 - t0)
        iters = r.iterations
        converged = r.converged
        residual = r.residual
    return float(np.median(times)), iters, converged, residual


def sweep(
    label: str,
    cats: np.ndarray,
    y: np.ndarray,
    *,
    use_precond: bool,
    tol: float,
    maxiter: int,
    local_sizes: list[int | None],
    seed: int,
) -> None:
    tag = "+precond" if use_precond else "-precond"
    full_label = f"{label} [{tag}]"
    baseline_iters: int | None = None
    baseline_time: float | None = None
    for ls in local_sizes:
        t, iters, converged, residual = run_one(
            cats,
            y,
            tol=tol,
            maxiter=maxiter,
            local_size=ls,
            seed=seed,
            use_precond=use_precond,
        )
        iter_cell = f"{iters}"
        time_cell = f"{t * 1e3:.1f}"
        if baseline_iters is not None and baseline_time is not None:
            iter_cell = f"{iters} ({iters - baseline_iters:+d})"
            time_cell = f"{t * 1e3:.1f} ({(t / baseline_time - 1) * 100:+.0f}%)"
        ls_cell = "off" if ls is None else str(ls)
        print(
            f"{full_label:<36} {ls_cell:>5} {iter_cell:>14} "
            f"{time_cell:>16} {residual:>11.2e} "
            f"{'yes' if converged else 'NO':>5}"
        )
        if ls is None:
            baseline_iters = iters
            baseline_time = t
    print()


def main() -> None:
    seed = 42
    # Tight tolerance pushes LSMR past the regime where the short recurrence
    # holds — that's where windowed MGS starts paying for itself.
    tol = 1e-14
    maxiter = 5000

    # Mix of well- and ill-conditioned problems.
    problems = [
        ("chain_3fe n=2000", "chain_3fe", {"n_levels": 2000}),
        ("chain_3fe n=5000", "chain_3fe", {"n_levels": 5000}),
        (
            "barbell_3fe n=2000 bw=1",
            "barbell_3fe",
            {"n_levels": 2000, "bridge_width": 1},
        ),
        (
            "barbell_3fe n=5000 bw=1",
            "barbell_3fe",
            {"n_levels": 5000, "bridge_width": 1},
        ),
        (
            "akm_disconnected 50 clusters",
            "akm_disconnected",
            {
                "n_workers": 10_000,
                "n_firms": 500,
                "n_years": 8,
                "n_clusters": 50,
                "within_mobility": 0.04,
                "cross_cluster_rate": 0.0005,
                "n_fe": 3,
            },
        ),
        (
            "clustered_3fe 50/50/50 c=8 br=2",
            "clustered_3fe",
            {
                "n_levels": (50, 50, 50),
                "n_clusters": 8,
                "obs_per_cluster": 200,
                "bridge_obs": 2,
            },
        ),
        (
            "akm extreme low mobility",
            "akm_low_mobility",
            {
                "n_workers": 5_000,
                "n_firms": 200,
                "n_years": 8,
                "annual_mobility_rate": 0.003,
                "n_fe": 3,
            },
        ),
        (
            "akm disconnected (10 clusters)",
            "akm_disconnected",
            {
                "n_workers": 5_000,
                "n_firms": 300,
                "n_years": 8,
                "n_clusters": 10,
                "within_mobility": 0.04,
                "cross_cluster_rate": 0.001,
                "n_fe": 3,
            },
        ),
    ]
    local_sizes: list[int | None] = [None, 5, 10, 20]

    header = (
        f"{'problem':<36} {'local':>5} {'iters':>14} "
        f"{'time_ms':>16} {'residual':>11} {'conv':>5}"
    )
    print(header)
    print("-" * len(header))

    for label, gen_name, params in problems:
        cats_list, _, y = get_generator(gen_name)(**params, seed=seed)
        cats = np.asfortranarray(np.column_stack(cats_list).astype(np.uint32))

        # Both regimes: with and without Schwarz preconditioner.
        sweep(
            label,
            cats,
            y,
            use_precond=True,
            tol=tol,
            maxiter=maxiter,
            local_sizes=local_sizes,
            seed=seed,
        )
        sweep(
            label,
            cats,
            y,
            use_precond=False,
            tol=tol,
            maxiter=maxiter,
            local_sizes=local_sizes,
            seed=seed,
        )


if __name__ == "__main__":
    main()
