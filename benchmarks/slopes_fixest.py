"""Varying-slopes reference benchmark: R fixest vs within (#65).

On-demand perf comparison for the varying-slopes epic (#64): times slope
estimation on shared synthetic DGPs and prints fit time + iteration counts
per tool. The R arm needs a provisioned R; the within arm always runs.

    pixi run -e fixest bench-slopes     # provisions R + fixest, runs the catalog
    pixi run bench-slopes               # within arm only

Edit ``SIZES`` to sweep scale.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ._problems import (
    _make_response,
    _reindex,
    find_largest_component,
    fixest_dgp,
    simulate_mobility,
    zipf_firm_sizes,
)

_R_WORKER = Path(__file__).with_name("fixest_fit.R")
SIZES = [100_000]  # observation counts to benchmark
REPEAT = 3  # timed fits per case (median reported)


@dataclass
class Case:
    """A varying-slopes estimation problem.

    ``fe_terms`` maps each factor to its slope columns, e.g.
    ``[("worker", ["z1"]), ("year", []), ("firm", [])]`` →
    ``worker[z1] + year + firm``. Both the R formula and the future within
    wiring derive from it; ``response``/``regressors`` tag the remaining
    columns by role.
    """

    name: str
    columns: dict[str, NDArray]
    response: str
    regressors: list[str]
    fe_terms: list[tuple[str, list[str]]]

    @property
    def n_obs(self) -> int:
        return len(self.columns[self.response])

    def formula(self) -> str:
        """The feols formula: ``y ~ x1 | worker[z1, z2] + year + firm``."""
        blocks = " + ".join(
            f"{f}[{', '.join(s)}]" if s else f for f, s in self.fe_terms
        )
        return f"{self.response} ~ {' + '.join(self.regressors)} | {blocks}"


def fixest_slope_case(n_obs: int, dgp_type: str, n_slopes: int) -> Case:
    """Fixest worker×year×firm panel; the worker factor carries the slopes.

    ``dgp_type``: ``"simple"`` (i.i.d. firms, well-connected — R's home turf)
    or ``"difficult"`` (band-structured firms, sparse coupling — within's
    sweet spot).
    """
    cats, n_levels, y = fixest_dgp(n_obs=n_obs, dgp_type=dgp_type, n_fe=3)
    rng = np.random.default_rng(43)
    n = len(y)
    worker = cats[0]

    x1 = rng.standard_normal(n)
    y = y + x1
    slopes: dict[str, NDArray] = {}
    for v in range(n_slopes):
        z = rng.standard_normal(n)
        gamma = 0.5 * rng.standard_normal(n_levels[0])
        y = y + z * gamma[worker]
        slopes[f"z{v + 1}"] = z

    factor_names = ["worker", "year", "firm"]
    columns: dict[str, NDArray] = {"y": y, "x1": x1, **slopes}
    columns.update({f: c.astype(np.int64) for f, c in zip(factor_names, cats)})
    fe_terms = [("worker", list(slopes)), ("year", []), ("firm", [])]
    return Case(f"fixest_{dgp_type}_v{n_slopes}", columns, "y", ["x1"], fe_terms)


def akm_slope_case(n_obs: int, slope_vars: str) -> Case:
    """AKM panel (Zipf firms, low mobility, clusters, largest component) with
    structured slopes derived from the mobility process:

    - ``experience`` (worker slope): entry cohort + calendar year.
    - ``tenure`` (firm slope): years since the worker's last move, reset on move.

    ``slope_vars``: ``"experience"``, ``"tenure"``, or ``"both"``.
    """
    if slope_vars not in ("experience", "tenure", "both"):
        raise ValueError(f"unknown slope_vars: {slope_vars!r}")
    with_experience = slope_vars in ("experience", "both")
    with_tenure = slope_vars in ("tenure", "both")

    n_years = 10
    n_workers = max(1, round(n_obs / n_years))
    n_firms = max(1, n_workers // 20)
    n_clusters = 5

    rng = np.random.default_rng(42)
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

    # Structured slopes on the (worker, year) grid.
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

    categories = [
        worker_ids.astype(np.int64),
        firm_ids.astype(np.int64),
        year_ids.astype(np.int64),
    ]
    n_levels = [int(c.max()) + 1 for c in categories]
    y = _make_response(categories, n_levels, rng)
    x1 = rng.standard_normal(len(y))
    y = y + x1
    if with_experience:
        gamma = 0.03 + 0.02 * rng.standard_normal(n_levels[0])
        y = y + gamma[worker_ids] * exp_obs
    if with_tenure:
        delta = 0.01 + 0.01 * rng.standard_normal(n_levels[1])
        y = y + delta[firm_ids] * ten_obs

    columns: dict[str, NDArray] = {
        "y": y,
        "x1": x1,
        "worker": categories[0],
        "firm": categories[1],
        "year": categories[2],
    }
    if with_experience:
        columns["exp"] = exp_obs
    if with_tenure:
        columns["ten"] = ten_obs
    fe_terms = [
        ("worker", ["exp"] if with_experience else []),
        ("firm", ["ten"] if with_tenure else []),
        ("year", []),
    ]
    return Case(f"akm_{slope_vars}", columns, "y", ["x1"], fe_terms)


# Each builder takes n_obs and returns a self-named Case.
CASES: list[Callable[[int], Case]] = [
    lambda n: fixest_slope_case(n, "simple", 1),
    lambda n: fixest_slope_case(n, "simple", 3),
    lambda n: fixest_slope_case(n, "difficult", 1),
    lambda n: fixest_slope_case(n, "difficult", 3),
    lambda n: akm_slope_case(n, "experience"),
    lambda n: akm_slope_case(n, "both"),
]


def _write_csv(path: str, columns: dict[str, NDArray]) -> None:
    """Write columns to CSV — integer factor codes as ints, the rest as %g."""
    names = list(columns)
    fmts = [
        "%d" if np.issubdtype(columns[c].dtype, np.integer) else "%.10g" for c in names
    ]
    matrix = np.column_stack([columns[c] for c in names])
    np.savetxt(
        path, matrix, delimiter=",", header=",".join(names), comments="", fmt=fmts
    )


def _r_available() -> bool:
    """True if Rscript is on PATH and the fixest package is installed."""
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        ["Rscript", "-e", 'quit(status = !requireNamespace("fixest", quietly = TRUE))'],
        capture_output=True,
    )
    return probe.returncode == 0


def run_fixest(case: Case) -> dict:
    """Time an feols fit of ``case`` via the R worker; parse RESULT_* markers."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        csv_path = f.name
    try:
        _write_csv(csv_path, case.columns)
        proc = subprocess.run(
            ["Rscript", str(_R_WORKER), csv_path, case.formula(), str(REPEAT)],
            capture_output=True,
            text=True,
        )
    finally:
        Path(csv_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed for {case.name}:\n{proc.stderr.strip()}")

    markers: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_"):
            key, _, value = line.partition(" ")
            markers[key] = value.strip()

    return {
        "tool": "R fixest",
        "case": case.name,
        "n_obs": case.n_obs,
        "time_s": float(markers["RESULT_TIME"]),
        "iters": markers.get("RESULT_ITERS", "?"),
    }


def run_within(case: Case) -> dict:
    """Time within on ``case``: one solver build per fit, then a demeaning
    solve for the response and each regressor (matching feols's per-variable
    iteration report)."""
    import within

    effects = [
        within.Effect(
            case.columns[factor].astype(np.uint32),
            True,
            [case.columns[s].astype(np.float64) for s in slopes],
        )
        for factor, slopes in case.fe_terms
    ]
    targets = [case.response, *case.regressors]

    times: list[float] = []
    iters: list[str] = []
    for _ in range(REPEAT):
        start = time.perf_counter()
        solver = within.Solver(effects)
        results = [solver.solve(case.columns[t].astype(np.float64)) for t in targets]
        times.append(time.perf_counter() - start)
        iters = [f"{r.iterations}{'' if r.converged else '!'}" for r in results]
    return {
        "tool": "within",
        "case": case.name,
        "n_obs": case.n_obs,
        "time_s": float(np.median(times)),
        "iters": "/".join(iters),
    }


def main() -> int:
    r_available = _r_available()
    if not r_available:
        print(
            "R fixest: skipped (Rscript/fixest not found) — "
            "run `pixi run -e fixest bench-slopes` to provision R."
        )

    rows: list[dict] = []
    for size in SIZES:
        for builder in CASES:
            case = builder(size)
            print(f"  {case.name}  n_obs={case.n_obs:,}  feols: {case.formula()}")
            if r_available:
                try:
                    rows.append(run_fixest(case))
                except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                    print(f"    R fixest failed: {exc}")
            try:
                rows.append(run_within(case))
            except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                print(f"    within failed: {exc}")

    print(f"\n{'tool':<10}{'case':<22}{'n_obs':>10}{'time_s':>9}   iters")
    for r in rows:
        print(
            f"{r['tool']:<10}{r['case']:<22}{r['n_obs']:>10,}"
            f"{r['time_s']:>9.3f}   {r['iters']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
