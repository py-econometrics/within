"""Full Schwarz vs reduced PairSchur/BlockJacobi (± rigid coarse) on the akm_slopes cells (probe, not a suite)."""

import os
import sys
import time

import numpy as np

from within import LsmrOptions, PreconditionerConfig, solve
from within._within import ReducedInverse, solve_reduced
from within.config import LocalSolverConfig

from benchmarks.suites.akm_slopes import _LADDER, _SPECS, _mover_panel, _slope_case

n_obs = int(sys.argv[1])
designs = sys.argv[2].split(",")
specs = sys.argv[3].split(",")
opts = LsmrOptions(tol=1e-12, maxiter=20_000)
for design in designs:
    panel = _mover_panel(n_obs, _LADDER[design], 42)
    for name, columns in _SPECS:
        if name not in specs:
            continue
        case = _slope_case(panel, columns, 42)
        row = [f"{design:14} {name:18}"]
        for label, run in [
            (
                "full",
                lambda: solve(
                    case.effects,
                    case.y,
                    options=opts,
                    preconditioner=PreconditionerConfig.Additive(),
                ),
            ),
            (
                "red_pairs",
                lambda: solve_reduced(
                    case.effects,
                    case.y,
                    options=opts,
                    local_solver=LocalSolverConfig(),
                    inverse=ReducedInverse.PairSchur,
                ),
            ),
            (
                "red_bj",
                lambda: solve_reduced(
                    case.effects,
                    case.y,
                    options=opts,
                    local_solver=LocalSolverConfig(),
                    inverse=ReducedInverse.BlockJacobi,
                ),
            ),
            (
                "red_pairs+rigid",
                lambda: solve_reduced(
                    case.effects,
                    case.y,
                    options=opts,
                    local_solver=LocalSolverConfig(),
                    inverse=ReducedInverse.PairSchur,
                ),
            ),
            (
                "red_bj+rigid",
                lambda: solve_reduced(
                    case.effects,
                    case.y,
                    options=opts,
                    local_solver=LocalSolverConfig(),
                    inverse=ReducedInverse.BlockJacobi,
                ),
            ),
        ]:
            if os.environ.get("ARMS") and label not in os.environ["ARMS"].split(","):
                continue
            if "rigid" in label:
                os.environ["WITHIN_REDUCED_RIGID"] = os.environ.get("RIGID_K", "150")
            else:
                os.environ.pop("WITHIN_REDUCED_RIGID", None)
            t = time.perf_counter()
            try:
                r = run()
                if len(row) == 1:
                    ref = np.asarray(r.demeaned)
                err = np.linalg.norm(np.asarray(r.demeaned) - ref) / np.linalg.norm(
                    case.y
                )
                row.append(
                    f"{label} {r.iterations:6}{'' if r.converged else '!'} {time.perf_counter() - t:6.1f}s e{err:.0e}"
                )
            except Exception as e:  # noqa: BLE001
                row.append(f"{label} ERR {e}")
        print(" | ".join(row), flush=True)
