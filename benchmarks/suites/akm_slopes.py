"""AKM sorting/mobility ladder crossed with how two factors' slope covariates relate.

The worker-firm mover graph is swept along the two axes the AKM literature parameterizes —
assortative sorting strength at full mobility, and the per-year move rate — and each panel is
solved under five slope specifications: none, independent covariates, one covariate shared by
both large factors, that covariate perturbed on one side, and the experience/tenure pair.
Sharing a covariate puts a null direction across both terms' intercept and slope channels, so
no factor-pair subdomain contains it. The last two arms sit just off that exact kernel by
different routes: an injected epsilon fixes the offset directly, while experience minus tenure
is constant within a job spell and differs only on movers, so mobility sets the offset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from within import Effect, LsmrOptions, solve

from .._framework import (
    BenchmarkResult,
    SlopeCase,
    SolverConfig,
    SuiteOptions,
    make_additive_schwarz,
    max_abs_group_mean,
    suite,
)
from .._problems import AssortativeMobility, _make_response, _reindex
from .._table import print_pivot, print_table

SlopeColumns = Callable[
    ["_MoverPanel", np.random.Generator],
    tuple[list[NDArray[np.float64]] | None, list[NDArray[np.float64]] | None],
]

NEAR_SHARED_EPS = 1e-3
# benchmark_lsmr's 1e-7 floor hides the penalty: shared_year is 23 iters there, 261 at 1e-12.
CONDITIONING_TOL = 1e-12


@dataclass(frozen=True)
class _MoverPanel:
    """One worker/firm/year panel and the covariates the slope specifications draw on:
    calendar year, experience (entry cohort + year), and tenure (years since the last move)."""

    codes: list[NDArray[np.uint32]]
    n_levels: list[int]
    year: NDArray[np.float64]
    experience: NDArray[np.float64]
    tenure: NDArray[np.float64]


_SPECS: list[tuple[str, SlopeColumns]] = [
    ("intercepts", lambda panel, rng: (None, None)),
    (
        "independent",
        lambda panel, rng: (
            [rng.standard_normal(len(panel.year))],
            [rng.standard_normal(len(panel.year))],
        ),
    ),
    ("shared_year", lambda panel, rng: ([panel.year], [panel.year])),
    (
        "near_shared_year",
        lambda panel, rng: (
            [panel.year],
            [panel.year + NEAR_SHARED_EPS * rng.standard_normal(len(panel.year))],
        ),
    ),
    ("experience_tenure", lambda panel, rng: ([panel.experience], [panel.tenure])),
]

_LADDER: dict[str, AssortativeMobility] = {
    "mobility=1.0": AssortativeMobility(mobility=1.0),
    "mobility=0.05": AssortativeMobility(mobility=0.05),
    "mobility=0.001": AssortativeMobility(mobility=0.001),
    "sorting=1e4": AssortativeMobility(mobility=1.0, sorting=10_000.0),
}


def _mover_panel(n_obs: int, design: AssortativeMobility, seed: int) -> _MoverPanel:
    """Simulate *design* at roughly *n_obs* rows, keeping every component as the paper does."""
    n_workers = max(1, n_obs // design.n_years)
    sized = replace(design, n_workers=n_workers, n_firms=max(2, n_workers // 2))
    rng = np.random.default_rng(seed)
    assignments = sized.simulate(rng)

    tenure = np.zeros((n_workers, sized.n_years))
    for year in range(1, sized.n_years):
        stayed = assignments[:, year] == assignments[:, year - 1]
        tenure[:, year] = (tenure[:, year - 1] + 1.0) * stayed
    entry_cohort = rng.integers(0, 30, size=n_workers)

    worker_ids = np.repeat(np.arange(n_workers, dtype=np.intp), sized.n_years)
    year_ids = np.tile(np.arange(sized.n_years, dtype=np.intp), n_workers)
    codes = [
        _reindex(ids).astype(np.uint32)
        for ids in (worker_ids, assignments.ravel(), year_ids)
    ]
    year = year_ids.astype(np.float64)
    return _MoverPanel(
        codes,
        [int(c.max()) + 1 for c in codes],
        year,
        entry_cohort[worker_ids] + year,
        tenure.ravel(),
    )


def _slope_case(panel: _MoverPanel, columns: SlopeColumns, seed: int) -> SlopeCase:
    """Apply one slope specification to *panel*, generating a matching response."""
    rng = np.random.default_rng(seed)
    y = _make_response(panel.codes, panel.n_levels, rng)
    worker_slopes, firm_slopes = columns(panel, rng)
    for factor, slopes in ((0, worker_slopes), (1, firm_slopes)):
        for z in slopes or ():
            gamma = 0.5 * rng.standard_normal(panel.n_levels[factor])
            y += z * gamma[panel.codes[factor]]

    effects = [
        Effect(panel.codes[0], True, worker_slopes),
        Effect(panel.codes[1], True, firm_slopes),
        Effect(panel.codes[2], True, None),
    ]
    return SlopeCase(effects, y, panel.codes, panel.n_levels)


@suite(
    "akm_slopes",
    description="AKM sorting/mobility ladder x shared vs independent slope covariates",
    tags=("slopes", "akm", "3fe", "conditioning"),
)
def run_akm_slopes(opts: SuiteOptions) -> list[BenchmarkResult]:
    n_obs_list = opts.select(
        smoke=[100_000], iterate=[100_000], full=[100_000, 1_000_000]
    )
    designs = opts.select(
        smoke=["mobility=1.0", "mobility=0.05"],
        iterate=["mobility=1.0", "mobility=0.05", "mobility=0.001"],
        full=list(_LADDER),
    )
    cfg = SolverConfig(
        "LSMR(Schwarz)",
        LsmrOptions(
            tol=min(opts.tol, CONDITIONING_TOL), maxiter=max(opts.maxiter, 20_000)
        ),
        preconditioner=make_additive_schwarz(local_solver=None),
    )

    results: list[BenchmarkResult] = []
    for n_obs in n_obs_list:
        for design_name in designs:
            panel = _mover_panel(n_obs, _LADDER[design_name], opts.seed)
            for spec_name, columns in _SPECS:
                case = _slope_case(panel, columns, opts.seed)
                name = f"n={n_obs:,} {design_name} {spec_name}"
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
