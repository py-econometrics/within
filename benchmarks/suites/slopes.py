"""Varying-slopes scaling suite.

Slope estimation on synthetic panels across scales, within-only (no external
baseline): a fixest-style worker×year×firm panel whose worker factor carries a
growing number of slopes, and AKM mobility panels with structured
experience/tenure slopes. Tracks iteration counts, per-iteration cost, and
convergence as slope count, topology, and scale vary — the axes on which
varying-slopes conditioning is known to bite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from within import Effect, solve

from .._framework import (
    BenchmarkResult,
    SolverConfig,
    SuiteOptions,
    benchmark_lsmr,
    make_additive_schwarz,
    max_abs_group_mean,
    suite,
)
from .._problems import (
    _make_response,
    _reindex,
    find_largest_component,
    fixest_dgp,
    simulate_mobility,
    zipf_firm_sizes,
)
from .._table import print_pivot, print_table


@dataclass(frozen=True)
class _SlopeCase:
    """A varying-slopes benchmark problem: the ``Effect`` design to solve, the
    response, and the bare factor codes the group-mean demean check needs."""

    effects: list[Effect]
    y: NDArray[np.float64]
    categories: list[NDArray[np.uint32]]
    n_levels: list[int]


def _fixest_slopes(n_obs: int, dgp_type: str, n_slopes: int, seed: int) -> _SlopeCase:
    """Worker×year×firm panel; the worker factor carries ``n_slopes`` slopes.

    ``dgp_type`` is ``"simple"`` (i.i.d. firms, well-connected) or
    ``"difficult"`` (band-structured firms, sparse coupling).
    """
    cats, n_levels, y = fixest_dgp(n_obs=n_obs, dgp_type=dgp_type, n_fe=3, seed=seed)
    rng = np.random.default_rng(seed + 1)
    worker = cats[0]
    slope_cols: list[NDArray[np.float64]] = []
    for _ in range(n_slopes):
        z = rng.standard_normal(len(y))
        gamma = 0.5 * rng.standard_normal(n_levels[0])
        y += z * gamma[worker]
        slope_cols.append(z)
    codes = [c.astype(np.uint32) for c in cats]
    effects = [
        Effect(codes[0], True, slope_cols),
        Effect(codes[1], True, None),
        Effect(codes[2], True, None),
    ]
    return _SlopeCase(effects, y, codes, n_levels)


def _akm_slopes(n_obs: int, slope_vars: str, seed: int) -> _SlopeCase:
    """AKM mobility panel (Zipf firms, clustered low mobility, largest component)
    with structured slopes derived from the mobility process:

    - ``experience`` (worker slope): entry cohort + calendar year.
    - ``tenure`` (firm slope): years since the worker's last move, reset on move.

    ``slope_vars`` is ``"experience"``, ``"tenure"``, or ``"both"``.
    """
    with_exp = slope_vars in ("experience", "both")
    with_ten = slope_vars in ("tenure", "both")

    n_years = 10
    n_workers = max(1, round(n_obs / n_years))
    n_firms = max(1, n_workers // 20)
    n_clusters = 5

    rng = np.random.default_rng(seed)
    firm_weights = zipf_firm_sizes(n_firms, 1.3)
    cluster_map = np.repeat(
        np.arange(n_clusters, dtype=np.intp),
        (n_firms + n_clusters - 1) // n_clusters,
    )[:n_firms]
    initial_firm = rng.choice(n_firms, size=n_workers, p=firm_weights).astype(np.intp)
    assignments = simulate_mobility(
        initial_firm,
        n_years,
        0.10,
        firm_weights,
        rng,
        cluster_map=cluster_map,
        cross_cluster_rate=0.02,
    )

    entry = rng.integers(0, 30, size=n_workers)
    experience = (entry[:, None] + np.arange(n_years)[None, :]).astype(np.float64)
    tenure = np.zeros((n_workers, n_years), dtype=np.float64)
    for t in range(1, n_years):
        stayed = assignments[:, t] == assignments[:, t - 1]
        tenure[:, t] = (tenure[:, t - 1] + 1.0) * stayed

    worker_ids = np.repeat(np.arange(n_workers, dtype=np.intp), n_years)
    year_ids = np.tile(np.arange(n_years, dtype=np.intp), n_workers)
    firm_ids = assignments.ravel().astype(np.intp)
    exp_obs = experience.ravel()
    ten_obs = tenure.ravel()

    keep = find_largest_component(worker_ids, firm_ids)
    worker_ids = _reindex(worker_ids[keep])
    firm_ids = _reindex(firm_ids[keep])
    year_ids = _reindex(year_ids[keep])
    exp_obs = exp_obs[keep]
    ten_obs = ten_obs[keep]

    codes = [
        worker_ids.astype(np.uint32),
        firm_ids.astype(np.uint32),
        year_ids.astype(np.uint32),
    ]
    n_levels = [int(c.max()) + 1 for c in codes]
    y = _make_response(codes, n_levels, rng)
    if with_exp:
        gamma = 0.03 + 0.02 * rng.standard_normal(n_levels[0])
        y += gamma[worker_ids] * exp_obs
    if with_ten:
        delta = 0.01 + 0.01 * rng.standard_normal(n_levels[1])
        y += delta[firm_ids] * ten_obs

    effects = [
        Effect(codes[0], True, [exp_obs] if with_exp else None),
        Effect(codes[1], True, [ten_obs] if with_ten else None),
        Effect(codes[2], True, None),
    ]
    return _SlopeCase(effects, y, codes, n_levels)


_CASES: list[tuple[str, Callable[[int, int], _SlopeCase]]] = [
    ("fixest_simple_v1", lambda n, s: _fixest_slopes(n, "simple", 1, s)),
    ("fixest_simple_v3", lambda n, s: _fixest_slopes(n, "simple", 3, s)),
    ("fixest_difficult_v1", lambda n, s: _fixest_slopes(n, "difficult", 1, s)),
    ("fixest_difficult_v3", lambda n, s: _fixest_slopes(n, "difficult", 3, s)),
    ("akm_experience", lambda n, s: _akm_slopes(n, "experience", s)),
    ("akm_both", lambda n, s: _akm_slopes(n, "both", s)),
]


@suite(
    "slopes",
    description="Varying-slopes scaling: fixest + AKM panels with per-level slopes",
    tags=("slopes", "akm", "3fe", "scaling"),
)
def run_slopes(opts: SuiteOptions) -> list[BenchmarkResult]:
    n_obs_list = opts.select(
        smoke=[100_000],
        iterate=[100_000, 1_000_000],
        full=[100_000, 1_000_000, 5_000_000],
    )
    cfg = SolverConfig(
        "LSMR(Schwarz)",
        benchmark_lsmr(opts),
        preconditioner=make_additive_schwarz(local_solver=None),
    )

    results: list[BenchmarkResult] = []
    for n_obs in n_obs_list:
        for case_name, builder in _CASES:
            case = builder(n_obs, opts.seed)
            name = f"n={n_obs:,} {case_name}"
            print(f"\n  {name}: Rows={len(case.y):,}")
            try:
                r = solve(
                    case.effects,
                    case.y,
                    options=cfg.config,
                    preconditioner=cfg.preconditioner,
                )
            except Exception as e:  # noqa: BLE001 - report and continue the sweep
                print(f"    WARNING: {cfg.label} failed: {e}")
                continue
            results.append(
                BenchmarkResult(
                    problem=name,
                    config=cfg.label,
                    n_dofs=len(r.x),
                    n_rows=len(case.y),
                    setup_time=r.time_setup,
                    solve_time=r.time_solve,
                    iterations=r.iterations,
                    final_residual=r.residual,
                    demeaning_error=max_abs_group_mean(
                        case.categories, case.n_levels, r.demeaned
                    ),
                    converged=r.converged,
                )
            )

    print_table(results)
    print_table(
        results,
        columns=["config", "setup_time", "solve_time", "iterations", "ms_per_iter"],
        title="Per-iteration cost",
    )
    print("\n")
    print_pivot(results)
    return results
