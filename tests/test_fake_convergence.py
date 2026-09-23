"""The akm_slopes cell whose reduced-CG scaling once drove the Schwarz factor near singular
(#286): the amplified local solve collapsed the LSMR recurrences into a converged report while
the true normal-equation residual stayed O(1) (#290). Single-threaded reductions reproduce it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CELL = dict(
    n_obs=100_000, design="mobility=0.05", spec="shared_year", seed=42, tol=1e-12
)


def solve_cell() -> dict[str, float | bool | int]:
    from benchmarks._problems import _make_response
    from benchmarks.suites.akm_slopes import _LADDER, _SPECS, _mover_panel, _slope_case
    from within import LsmrOptions, PreconditionerConfig, solve

    panel = _mover_panel(CELL["n_obs"], _LADDER[CELL["design"]], CELL["seed"])
    columns = dict(_SPECS)[CELL["spec"]]
    case = _slope_case(panel, columns, CELL["seed"])
    # Replay the response draw so the slope covariates come out as the case saw them.
    rng = np.random.default_rng(CELL["seed"])
    _make_response(panel.codes, panel.n_levels, rng)
    slopes = columns(panel, rng)

    result = solve(
        case.effects,
        case.y,
        options=LsmrOptions(tol=CELL["tol"], maxiter=20_000),
        preconditioner=PreconditionerConfig.Additive(),
    )

    def group_mean(factor: int, z: np.ndarray | None) -> float:
        codes, n = case.categories[factor], case.n_levels[factor]
        weighted = result.demeaned if z is None else z * result.demeaned
        sums, counts = np.zeros(n), np.zeros(n)
        np.add.at(sums, codes, weighted)
        np.add.at(counts, codes, 1.0)
        return float(np.abs(sums / np.maximum(counts, 1.0)).max())

    worst = max(group_mean(f, None) for f in range(3))
    for factor, columns_f in enumerate(slopes[:2]):
        for z in columns_f or ():
            worst = max(worst, group_mean(factor, z))
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "worst_group_mean": worst,
    }


def test_shared_year_cell_solves_honestly_on_one_thread():
    env = {
        **os.environ,
        "RAYON_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["converged"], report
    assert report["worst_group_mean"] < 1e-6, report


if __name__ == "__main__":
    print(json.dumps(solve_cell()))
