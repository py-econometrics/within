"""Stress matrix for the LSMR-only redesign decision.

Covers the gaps the existing benchmark suites cannot reach:
  - tight tolerance (1e-12, 1e-14) below the framework's 1e-7 floor
  - weighted problems (lognormal weights — heavy-tailed, realistic)
  - batch RHS via solve_batch (parallel inner-loop solves)
  - forced non-convergence (maxiter cap)
  - cross-method coefficient agreement at the API level
  - determinism across re-runs of the same (problem, seed)

Output: a PASS / WARN / FAIL verdict per (scenario, method, tolerance) cell
plus per-cell quality numbers (`ne_resid`, `demean_err`, `dx_vs_ref`).

Run from the project root:

    PYTHONPATH=. pixi run python scripts/krylov_stress_matrix.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from benchmarks._problems import get_generator
from within import (
    CG,
    GMRES,
    LSMR,
    AdditiveSchwarz,
    MultiplicativeSchwarz,
    Solver,
    solve_batch,
)
from within._within import ApproxCholConfig, ApproxSchurConfig, SchurComplement

sys.stdout.reconfigure(line_buffering=True)


SEED = 42
MAXITER = 10_000
N_REPEATS = 2


@dataclass
class Cell:
    scenario: str
    tol: float
    method: str
    iters: int
    converged: bool
    reported: float
    ne_resid: float
    demean_err: float
    dx_vs_ref: float
    t_solve: float


# ----------------------------------------------------------------------
# Helpers shared with krylov_method_comparison.py
# ----------------------------------------------------------------------
def schur(seed: int) -> SchurComplement:
    return SchurComplement(
        approx_chol=ApproxCholConfig(seed=seed),
        approx_schur=ApproxSchurConfig(seed=seed),
    )


def stack_categories(cats):
    arr = np.column_stack([c.astype(np.uint32) for c in cats])
    return np.asfortranarray(arr)


def factor_offsets(cats_2d):
    n_levels = [int(cats_2d[:, q].max()) + 1 for q in range(cats_2d.shape[1])]
    offs = [0]
    for n in n_levels:
        offs.append(offs[-1] + n)
    return offs


def D_x(cats_2d, x, offs):
    out = np.zeros(cats_2d.shape[0])
    for q in range(cats_2d.shape[1]):
        out += x[offs[q] : offs[q + 1]][cats_2d[:, q]]
    return out


def DT_v(cats_2d, v, offs, n_dofs):
    out = np.zeros(n_dofs)
    for q in range(cats_2d.shape[1]):
        np.add.at(out, offs[q] + cats_2d[:, q], v)
    return out


def true_ne_residual(cats_2d, y, x, offs, n_dofs, w=None):
    """||D^T W (y - D x)|| / ||D^T W y||."""
    r = y - D_x(cats_2d, x, offs)
    if w is not None:
        r = w * r
        wy = w * y
    else:
        wy = y
    DTr = DT_v(cats_2d, r, offs, n_dofs)
    DTy = DT_v(cats_2d, wy, offs, n_dofs)
    return float(np.linalg.norm(DTr) / max(np.linalg.norm(DTy), 1e-30))


def max_abs_group_mean(cats_2d, demeaned, w=None):
    worst = 0.0
    for q in range(cats_2d.shape[1]):
        cats = cats_2d[:, q]
        n_levels = int(cats.max()) + 1
        sums = np.zeros(n_levels)
        counts = np.zeros(n_levels)
        if w is None:
            np.add.at(sums, cats, demeaned)
            np.add.at(counts, cats, 1.0)
        else:
            np.add.at(sums, cats, w * demeaned)
            np.add.at(counts, cats, w)
        means = sums / np.maximum(counts, 1e-30)
        worst = max(worst, float(np.abs(means).max()))
    return worst


# ----------------------------------------------------------------------
# Scenario builders
# ----------------------------------------------------------------------
def base_problem():
    """fixest_dgp 1M, difficult, 3-FE — same as comparison script."""
    gen = get_generator("fixest_dgp")
    cats_list, n_levels, y = gen(
        n_obs=1_000_000, dgp_type="difficult", n_fe=3, seed=SEED
    )
    cats_2d = stack_categories(cats_list)
    offs = factor_offsets(cats_2d)
    n_dofs = sum(int(cats_2d[:, q].max()) + 1 for q in range(cats_2d.shape[1]))
    return cats_2d, y, offs, n_dofs


def lognormal_weights(n_obs: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.exp(rng.normal(loc=0.0, scale=sigma, size=n_obs))
    return w / w.mean()


def configs(tol: float, maxiter: int, seed: int):
    return [
        ("CG", CG(tol=tol, maxiter=maxiter), AdditiveSchwarz(local_solver=schur(seed))),
        (
            "GMRES",
            GMRES(tol=tol, maxiter=maxiter),
            MultiplicativeSchwarz(local_solver=schur(seed)),
        ),
        (
            "LSMR",
            LSMR(tol=tol, maxiter=maxiter),
            AdditiveSchwarz(local_solver=schur(seed)),
        ),
    ]


def run_solver(cats_2d, y, label, cfg, pc, weights=None) -> tuple[float, object]:
    Solver(cats_2d, config=cfg, preconditioner=pc, weights=weights)  # warmup
    times = []
    last = None
    for _ in range(N_REPEATS):
        s = Solver(cats_2d, config=cfg, preconditioner=pc, weights=weights)
        t0 = time.perf_counter()
        last = s.solve(y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), last


# ----------------------------------------------------------------------
# Pass/fail evaluation — TODO(human)
# ----------------------------------------------------------------------
def evaluate_verdict(
    cell: Cell, tol: float, ref_method: str = "LSMR"
) -> tuple[str, str]:
    """Decide PASS / WARN / FAIL for a solver result."""
    # Maxiter scenario: hitting the cap unconverged is the expected outcome.
    if cell.scenario.startswith("maxiter="):
        return ("WARN", "hit_maxiter") if not cell.converged else ("PASS", "ok")

    if not cell.converged:
        return ("FAIL", "not_converged")

    ne_floor = max(10 * tol, 1e-13)
    demean_warn = max(100 * tol, 1e-4)
    demean_fail = 1e-2
    dx_warn = max(100 * tol, 1e-4)
    dx_fail = 1e-2

    if cell.ne_resid > 1e-3:
        return ("FAIL", f"ne_resid={cell.ne_resid:.1e}")
    if cell.demean_err > demean_fail:
        return ("FAIL", f"demean={cell.demean_err:.1e}")
    if cell.method != ref_method and cell.dx_vs_ref == cell.dx_vs_ref:
        if cell.dx_vs_ref > dx_fail:
            return ("FAIL", f"dx_vs_{ref_method}={cell.dx_vs_ref:.1e}")
        if cell.dx_vs_ref > dx_warn:
            return ("WARN", f"dx_vs_{ref_method}={cell.dx_vs_ref:.1e}")
    if cell.demean_err > demean_warn:
        return ("WARN", f"demean={cell.demean_err:.1e}")
    if cell.ne_resid > ne_floor:
        return ("WARN", f"ne_resid={cell.ne_resid:.1e}")
    return ("PASS", "ok")


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------
def run_scenario(
    name: str,
    cats_2d,
    y,
    offs,
    n_dofs,
    weights,
    tols,
    maxiter,
    ref_method: str = "LSMR",
) -> list[Cell]:
    """Run all 3 methods at all tolerances; return per-cell results."""
    cells: list[Cell] = []
    for tol in tols:
        # First pass: gather all 3 results to compute dx_vs_ref later
        run_data: dict[str, tuple[float, object, np.ndarray]] = {}
        for label, cfg, pc in configs(tol, maxiter, SEED):
            try:
                t_solve, res = run_solver(cats_2d, y, label, cfg, pc, weights=weights)
                x = np.asarray(res.x, dtype=np.float64)
                run_data[label] = (t_solve, res, x)
            except Exception as e:
                print(f"  {name} tol={tol} {label} EXCEPTION: {e}")

        ref_x = run_data.get(ref_method, (None, None, None))[2]
        for label, (t_solve, res, x) in run_data.items():
            dx = (
                float(np.linalg.norm(x - ref_x) / max(np.linalg.norm(ref_x), 1e-30))
                if ref_x is not None
                else float("nan")
            )
            cells.append(
                Cell(
                    scenario=name,
                    tol=tol,
                    method=label,
                    iters=res.iterations,
                    converged=res.converged,
                    reported=res.residual,
                    ne_resid=true_ne_residual(cats_2d, y, x, offs, n_dofs, w=weights),
                    demean_err=max_abs_group_mean(
                        cats_2d, y - D_x(cats_2d, x, offs), w=weights
                    ),
                    dx_vs_ref=dx,
                    t_solve=t_solve,
                )
            )
    return cells


def run_batch_scenario(
    name: str,
    cats_2d,
    y,
    offs,
    n_dofs,
    n_rhs: int,
    tol: float,
    maxiter: int,
    ref_method: str = "LSMR",
) -> list[Cell]:
    """solve_batch path: stack n_rhs response columns and check per-column quality."""
    rng = np.random.default_rng(SEED + 1000)
    Y = np.column_stack(
        [y] + [rng.normal(0.0, 1.0, size=y.shape) for _ in range(n_rhs - 1)]
    )

    cells: list[Cell] = []
    run_data: dict[str, np.ndarray] = {}
    for label, cfg, pc in configs(tol, maxiter, SEED):
        try:
            res = solve_batch(cats_2d, Y, config=cfg, preconditioner=pc)
            X = np.asarray(res.x, dtype=np.float64)  # shape (n_dofs, n_rhs)
            run_data[label] = X
            # Aggregate quality across columns: worst column reported
            worst_demean = 0.0
            worst_ne = 0.0
            for k in range(n_rhs):
                x_k = X[:, k]
                y_k = Y[:, k]
                worst_ne = max(
                    worst_ne, true_ne_residual(cats_2d, y_k, x_k, offs, n_dofs)
                )
                worst_demean = max(
                    worst_demean,
                    max_abs_group_mean(cats_2d, y_k - D_x(cats_2d, x_k, offs)),
                )

            iters = (
                int(np.max(res.iterations))
                if hasattr(res, "iterations") and res.iterations is not None
                else 0
            )
            converged = (
                bool(np.all(res.converged))
                if hasattr(res, "converged") and res.converged is not None
                else True
            )
            cells.append(
                Cell(
                    scenario=name,
                    tol=tol,
                    method=label,
                    iters=iters,
                    converged=converged,
                    reported=0.0,  # batch doesn't expose a single residual
                    ne_resid=worst_ne,
                    demean_err=worst_demean,
                    dx_vs_ref=float("nan"),  # filled below
                    t_solve=0.0,
                )
            )
        except Exception as e:
            print(f"  {name} {label} EXCEPTION: {e}")

    # Fill dx_vs_ref using the first column (treat as representative)
    ref_X = run_data.get(ref_method)
    if ref_X is not None:
        ref0 = ref_X[:, 0]
        for c in cells:
            X = run_data.get(c.method)
            if X is not None:
                c.dx_vs_ref = float(
                    np.linalg.norm(X[:, 0] - ref0) / max(np.linalg.norm(ref0), 1e-30)
                )
    return cells


def run_determinism_scenario(cats_2d, y, offs, n_dofs, tol: float) -> list[Cell]:
    """Same problem, same seed, fresh build twice — measures threading nondeterminism."""
    cells: list[Cell] = []
    for label, cfg, pc in configs(tol, MAXITER, SEED):
        try:
            _, res1 = run_solver(cats_2d, y, label, cfg, pc)
            x1 = np.asarray(res1.x, dtype=np.float64)
            # Recreate config + preconditioner with same seed and re-solve
            for _, cfg2, pc2 in configs(tol, MAXITER, SEED):
                pass  # placeholder so each iteration uses fresh objects
            # Run again with fresh-but-same-seed configs
            _, cfg2, pc2 = next(c for c in configs(tol, MAXITER, SEED) if c[0] == label)
            _, res2 = run_solver(cats_2d, y, label, cfg2, pc2)
            x2 = np.asarray(res2.x, dtype=np.float64)
            dx = float(np.linalg.norm(x2 - x1) / max(np.linalg.norm(x1), 1e-30))
            cells.append(
                Cell(
                    scenario="determinism",
                    tol=tol,
                    method=label,
                    iters=res1.iterations,
                    converged=res1.converged,
                    reported=res1.residual,
                    ne_resid=true_ne_residual(cats_2d, y, x1, offs, n_dofs),
                    demean_err=max_abs_group_mean(cats_2d, y - D_x(cats_2d, x1, offs)),
                    dx_vs_ref=dx,  # repurpose dx_vs_ref to mean "x2 vs x1"
                    t_solve=0.0,
                )
            )
        except Exception as e:
            print(f"  determinism {label} EXCEPTION: {e}")
    return cells


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def fmt_e(x: float) -> str:
    return "—" if x != x else f"{x:.2e}"  # NaN check via x!=x


def print_matrix(cells: list[Cell]) -> tuple[int, int, int]:
    """Print the verdict matrix; return (n_pass, n_warn, n_fail)."""
    print(
        f"\n{'scenario':35s} {'tol':>9s}  {'method':<6s}  "
        f"{'iters':>5s}  {'conv':>4s}  {'ne_resid':>10s}  "
        f"{'demean':>10s}  {'dx_vs_ref':>10s}  {'t(s)':>7s}  verdict  reason"
    )
    print("-" * 130)
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for c in cells:
        verdict, reason = evaluate_verdict(c, c.tol)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(
            f"{c.scenario:35s} {c.tol:9.0e}  {c.method:<6s}  "
            f"{c.iters:5d}  {('Y' if c.converged else 'N'):>4s}  "
            f"{fmt_e(c.ne_resid):>10s}  {fmt_e(c.demean_err):>10s}  "
            f"{fmt_e(c.dx_vs_ref):>10s}  {c.t_solve:7.3f}  "
            f"{verdict:7s}  {reason}"
        )
    return counts["PASS"], counts["WARN"], counts["FAIL"]


def main():
    print("# LSMR-only redesign stress matrix")
    print(f"# seed={SEED}, maxiter={MAXITER}, repeats={N_REPEATS}")
    print()

    # Build base problem once; reuse for all unweighted scenarios
    print("Loading base problem (fixest_dgp 1M, difficult, 3-FE) ...")
    cats_2d, y, offs, n_dofs = base_problem()
    print(f"  n_obs={cats_2d.shape[0]:,}, n_dofs={n_dofs:,}")

    all_cells: list[Cell] = []

    # --- Scenario 1: unweighted, tight tolerance ladder
    print("\n[1] tolerance ladder (unweighted)")
    all_cells.extend(
        run_scenario(
            "unweighted fixest 1M",
            cats_2d,
            y,
            offs,
            n_dofs,
            weights=None,
            tols=(1e-4, 1e-8, 1e-12, 1e-14),
            maxiter=MAXITER,
        )
    )

    # --- Scenario 2: WLS with mild lognormal weights
    print("\n[2] WLS — lognormal weights sigma=0.5 (mild)")
    w_mild = lognormal_weights(cats_2d.shape[0], sigma=0.5, seed=SEED + 1)
    all_cells.extend(
        run_scenario(
            "WLS mild fixest 1M",
            cats_2d,
            y,
            offs,
            n_dofs,
            weights=w_mild,
            tols=(1e-8, 1e-12),
            maxiter=MAXITER,
        )
    )

    # --- Scenario 3: WLS with extreme lognormal weights
    print("\n[3] WLS — lognormal weights sigma=2.0 (heavy-tailed)")
    w_heavy = lognormal_weights(cats_2d.shape[0], sigma=2.0, seed=SEED + 2)
    all_cells.extend(
        run_scenario(
            "WLS heavy fixest 1M",
            cats_2d,
            y,
            offs,
            n_dofs,
            weights=w_heavy,
            tols=(1e-8, 1e-12),
            maxiter=MAXITER,
        )
    )

    # --- Scenario 4: forced non-convergence
    print("\n[4] maxiter=10 (forced non-convergence)")
    all_cells.extend(
        run_scenario(
            "maxiter=10 fixest 1M",
            cats_2d,
            y,
            offs,
            n_dofs,
            weights=None,
            tols=(1e-12,),
            maxiter=10,
        )
    )

    # --- Scenario 5: batch RHS
    print("\n[5] solve_batch with 5 RHS")
    all_cells.extend(
        run_batch_scenario(
            "batch x5 fixest 1M",
            cats_2d,
            y,
            offs,
            n_dofs,
            n_rhs=5,
            tol=1e-8,
            maxiter=MAXITER,
        )
    )

    # --- Scenario 6: determinism
    print("\n[6] determinism (same seed, two builds)")
    all_cells.extend(run_determinism_scenario(cats_2d, y, offs, n_dofs, tol=1e-12))

    # --- Verdict matrix
    print("\n## Verdict matrix")
    n_pass, n_warn, n_fail = print_matrix(all_cells)
    total = n_pass + n_warn + n_fail
    print()
    print(f"Summary: {n_pass}/{total} PASS, {n_warn} WARN, {n_fail} FAIL")
    if n_fail == 0:
        print("VERDICT: LSMR can replace CG/GMRES safely across this matrix.")
    else:
        print("VERDICT: investigate FAIL cells before committing to LSMR-only.")


if __name__ == "__main__":
    main()
