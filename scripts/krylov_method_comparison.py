"""Illustrative comparison of CG, GMRES, and LSMR for the within solver.

Produces the empirical evidence backing the LSMR-only redesign discussion:

  1. CG vs LSMR are at parity in iteration count and wall-clock time
     (~10% iteration overhead for LSMR, parity in wall-clock at scale).
  2. CG and LSMR agree on the solution to ~10 digits; GMRES with
     multiplicative Schwarz disagrees by 0.7%-3.5% on rank-deficient
     panels (every realistic econometric panel has rank deficiency).
  3. At loose tolerance (1e-8), LSMR's demean residual is ~10x tighter
     than CG's; at tight tolerance (1e-12) all methods produce the
     same demean residual.

Run from the project root:

    PYTHONPATH=. pixi run python scripts/krylov_method_comparison.py

The script bypasses the benchmark framework so it can configure
``tol=1e-12`` (the framework clamps to a 1e-7 floor).
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
)
from within._within import ApproxCholConfig, ApproxSchurConfig, SchurComplement

# Line-buffer stdout so progress is visible even when piped through tee/file.
sys.stdout.reconfigure(line_buffering=True)

SEEDS = (42, 123, 7)  # data + preconditioner seed varied across runs
MAXITER = 10_000
TOLERANCES = (1e-8, 1e-12)
N_REPEATS = 3  # timed measurements; one additional warmup pass precedes them

PROBLEMS = [
    ("fixest_dgp", {"n_obs": 1_000_000, "dgp_type": "difficult", "n_fe": 3}),
    ("fixest_dgp", {"n_obs": 5_000_000, "dgp_type": "difficult", "n_fe": 3}),
    ("akm_realistic", {"n_workers": 50_000, "n_firms": 2_000, "n_years": 15}),
    (
        "akm_disconnected",
        {
            "n_workers": 50_000,
            "n_firms": 2_000,
            "n_years": 15,
            "n_clusters": 10,
            "cross_cluster_rate": 0.001,
        },
    ),
]


@dataclass
class Run:
    label: str
    tol: float
    iters: int
    converged: bool
    reported: float
    ne_resid: float
    demean_err: float
    t_setup: float
    t_solve: float
    t_total: float
    x: np.ndarray


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


def true_ne_residual(cats_2d, y, x, offs, n_dofs):
    """||D^T (y - D x)|| / ||D^T y||."""
    r = y - D_x(cats_2d, x, offs)
    DTr = DT_v(cats_2d, r, offs, n_dofs)
    DTy = DT_v(cats_2d, y, offs, n_dofs)
    return float(np.linalg.norm(DTr) / max(np.linalg.norm(DTy), 1e-30))


def max_abs_group_mean(cats_2d, demeaned):
    """Max absolute per-level mean of demeaned residuals.

    If demeaning is perfect, the mean of `y - Dx` within every level of
    every factor is zero. This returns the worst-case violation across
    all (factor, level) pairs.
    """
    worst = 0.0
    for q in range(cats_2d.shape[1]):
        cats = cats_2d[:, q]
        n_levels = int(cats.max()) + 1
        sums = np.zeros(n_levels)
        counts = np.zeros(n_levels)
        np.add.at(sums, cats, demeaned)
        np.add.at(counts, cats, 1.0)
        means = sums / np.maximum(counts, 1.0)
        worst = max(worst, float(np.abs(means).max()))
    return worst


def demean_err(cats_2d, y, x, offs):
    return max_abs_group_mean(cats_2d, y - D_x(cats_2d, x, offs))


def configs(tol: float, seed: int):
    return [
        ("CG", CG(tol=tol, maxiter=MAXITER), AdditiveSchwarz(local_solver=schur(seed))),
        (
            "GMRES",
            GMRES(tol=tol, maxiter=MAXITER),
            MultiplicativeSchwarz(local_solver=schur(seed)),
        ),
        (
            "LSMR",
            LSMR(tol=tol, maxiter=MAXITER),
            AdditiveSchwarz(local_solver=schur(seed)),
        ),
    ]


def run_one(cats_2d, y, label, cfg, precond, tol, offs, n_dofs) -> Run:
    """Run with one warmup + N_REPEATS timed measurements; report median timings."""
    # Warmup
    Solver(cats_2d, config=cfg, preconditioner=precond)

    setups = []
    solves = []
    last_res = None
    last_x = None
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        solver = Solver(cats_2d, config=cfg, preconditioner=precond)
        t1 = time.perf_counter()
        res = solver.solve(y)
        t2 = time.perf_counter()
        setups.append(t1 - t0)
        solves.append(t2 - t1)
        last_res = res
        last_x = np.asarray(res.x, dtype=np.float64)

    t_setup = float(np.median(setups))
    t_solve = float(np.median(solves))
    return Run(
        label=label,
        tol=tol,
        iters=last_res.iterations,
        converged=last_res.converged,
        reported=last_res.residual,
        ne_resid=true_ne_residual(cats_2d, y, last_x, offs, n_dofs),
        demean_err=demean_err(cats_2d, y, last_x, offs),
        t_setup=t_setup,
        t_solve=t_solve,
        t_total=t_setup + t_solve,
        x=last_x,
    )


def fmt_e(x: float) -> str:
    return f"{x:.2e}"


def print_problem_block(name, n_obs, n_dofs, runs_by_tol):
    print(f"\n### {name}  (n_obs={n_obs:,}, n_dofs={n_dofs:,})\n")
    print(
        "| tol | method | iters | reported | ne_resid | demean_err | "
        "t_setup(s) | t_solve(s) | t_total(s) | ms/iter | dx_vs_cg | dx_vs_lsmr |"
    )
    print("|" + "|".join(["---"] * 12) + "|")
    for tol, runs in runs_by_tol.items():
        cg = next((r for r in runs if r.label == "CG"), None)
        ls = next((r for r in runs if r.label == "LSMR"), None)
        for r in runs:
            dx_cg = (
                np.linalg.norm(r.x - cg.x) / max(np.linalg.norm(cg.x), 1e-30)
                if cg is not None
                else float("nan")
            )
            dx_ls = (
                np.linalg.norm(r.x - ls.x) / max(np.linalg.norm(ls.x), 1e-30)
                if ls is not None
                else float("nan")
            )
            ms_iter = 1000.0 * r.t_solve / max(r.iters, 1)
            print(
                f"| {tol:.0e} | {r.label} | {r.iters} | "
                f"{fmt_e(r.reported)} | {fmt_e(r.ne_resid)} | "
                f"{fmt_e(r.demean_err)} | "
                f"{r.t_setup:.4f} | {r.t_solve:.4f} | {r.t_total:.4f} | "
                f"{ms_iter:.2f} | "
                f"{fmt_e(dx_cg)} | {fmt_e(dx_ls)} |"
            )


def print_timing_summary(all_runs):
    """Cross-problem timing summary table: t_setup, t_solve, t_total per method."""
    print("\n## Timing summary (all problems)\n")
    print("| problem | tol | metric | CG | GMRES | LSMR |")
    print("|" + "|".join(["---"] * 6) + "|")
    for (problem, tol), runs in all_runs.items():
        by_label = {r.label: r for r in runs}
        if not all(k in by_label for k in ("CG", "GMRES", "LSMR")):
            continue
        for metric in ("t_setup", "t_solve", "t_total"):
            cg_v = getattr(by_label["CG"], metric)
            gm_v = getattr(by_label["GMRES"], metric)
            ls_v = getattr(by_label["LSMR"], metric)
            print(
                f"| {problem} | {tol:.0e} | {metric}(s) | "
                f"{cg_v:.4f} | {gm_v:.4f} | {ls_v:.4f} |"
            )


def print_seed_robustness_summary(per_seed_runs):
    """Cross-seed summary: shows dx_vs_cg / dx_vs_lsmr range across seeds.

    per_seed_runs: dict[(problem, tol)] -> dict[seed] -> list[Run]
    """
    print("\n## Seed robustness — `dx_vs_cg` for GMRES across seeds\n")
    print("| problem | tol | " + " | ".join(f"seed={s}" for s in SEEDS) + " |")
    print("|" + "|".join(["---"] * (2 + len(SEEDS))) + "|")
    for (problem, tol), per_seed in per_seed_runs.items():
        cells = []
        for s in SEEDS:
            runs = per_seed.get(s, [])
            cg = next((r for r in runs if r.label == "CG"), None)
            gm = next((r for r in runs if r.label == "GMRES"), None)
            if cg is None or gm is None:
                cells.append("—")
                continue
            dx = np.linalg.norm(gm.x - cg.x) / max(np.linalg.norm(cg.x), 1e-30)
            cells.append(f"{dx:.2e}")
        print(f"| {problem} | {tol:.0e} | " + " | ".join(cells) + " |")

    print("\n## Seed robustness — `dx_vs_cg` for LSMR across seeds\n")
    print("| problem | tol | " + " | ".join(f"seed={s}" for s in SEEDS) + " |")
    print("|" + "|".join(["---"] * (2 + len(SEEDS))) + "|")
    for (problem, tol), per_seed in per_seed_runs.items():
        cells = []
        for s in SEEDS:
            runs = per_seed.get(s, [])
            cg = next((r for r in runs if r.label == "CG"), None)
            ls = next((r for r in runs if r.label == "LSMR"), None)
            if cg is None or ls is None:
                cells.append("—")
                continue
            dx = np.linalg.norm(ls.x - cg.x) / max(np.linalg.norm(cg.x), 1e-30)
            cells.append(f"{dx:.2e}")
        print(f"| {problem} | {tol:.0e} | " + " | ".join(cells) + " |")

    print("\n## Seed robustness — iteration counts across seeds\n")
    print("| problem | tol | method | " + " | ".join(f"seed={s}" for s in SEEDS) + " |")
    print("|" + "|".join(["---"] * (3 + len(SEEDS))) + "|")
    for (problem, tol), per_seed in per_seed_runs.items():
        for label in ("CG", "GMRES", "LSMR"):
            cells = []
            for s in SEEDS:
                runs = per_seed.get(s, [])
                r = next((rr for rr in runs if rr.label == label), None)
                cells.append(str(r.iters) if r is not None else "—")
            print(f"| {problem} | {tol:.0e} | {label} | " + " | ".join(cells) + " |")


def main():
    print("# Krylov method comparison: CG vs GMRES(Mult-Schwarz) vs LSMR(Schwarz)\n")
    print(
        f"Seeds: {SEEDS}, maxiter: {MAXITER}, tolerances: {TOLERANCES}, "
        f"repeats: {N_REPEATS} (median reported per seed)\n"
    )
    print(
        "All methods use the same Schur-complement local solver "
        "(approximate Cholesky + approximate Schur)."
    )
    print(
        "\nTimings: t_setup = Solver construction (preconditioner build), "
        "t_solve = solver.solve() call, t_total = t_setup + t_solve. "
        f"One warmup pass precedes {N_REPEATS} timed measurements; "
        "median of the timed runs is reported."
    )
    print(
        f"\nThe per-problem tables below are reported for seed={SEEDS[0]}; "
        "the seed-robustness summary at the end shows how the headline "
        "metrics vary across all seeds.\n"
    )

    all_runs: dict[tuple[str, float], list[Run]] = {}
    per_seed_runs: dict[tuple[str, float], dict[int, list[Run]]] = {}

    for seed in SEEDS:
        if len(SEEDS) > 1:
            print("\n========================================================")
            print(f"# Seed = {seed}")
            print("========================================================\n")

        for prob_name, params in PROBLEMS:
            gen = get_generator(prob_name)
            cats_list, n_levels, y = gen(**params, seed=seed)
            cats_2d = stack_categories(cats_list)
            offs = factor_offsets(cats_2d)
            n_dofs = sum(int(cats_2d[:, q].max()) + 1 for q in range(cats_2d.shape[1]))
            n_obs = cats_2d.shape[0]

            runs_by_tol: dict[float, list[Run]] = {}
            title_short = prob_name + (f"({params})" if params else "")
            for tol in TOLERANCES:
                runs = []
                for label, cfg, pc in configs(tol, seed):
                    try:
                        runs.append(
                            run_one(cats_2d, y, label, cfg, pc, tol, offs, n_dofs)
                        )
                    except Exception as e:
                        print(
                            f"  {prob_name} tol={tol} seed={seed}: {label} FAILED ({e})"
                        )
                runs_by_tol[tol] = runs
                if seed == SEEDS[0]:
                    all_runs[(title_short, tol)] = runs
                per_seed_runs.setdefault((title_short, tol), {})[seed] = runs

            if seed == SEEDS[0]:
                title = f"{prob_name} {params}" if params else prob_name
                print_problem_block(title, n_obs, n_dofs, runs_by_tol)

    print_timing_summary(all_runs)
    print_seed_robustness_summary(per_seed_runs)


if __name__ == "__main__":
    main()
