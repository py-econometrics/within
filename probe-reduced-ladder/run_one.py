"""One solve of one panel under one config; prints a JSON line."""

import json
import os
import resource
import sys
import time

import numpy as np
import within as w
import within._within as m

panel, config = sys.argv[1], sys.argv[2]
d = np.load(panel)
n_terms = sum(1 for k in d.files if k.startswith("codes_"))
effects = []
for i in range(n_terms):
    slopes = [d[k] for k in sorted(d.files) if k.startswith(f"slopes_{i}_")]
    effects.append(w.Effect(d[f"codes_{i}"], True, slopes or None))
weights = d["weights"] if "weights" in d.files else None
y = d["y"]

P = w.PreconditionerConfig
lazy = m.CoarseCorrection(on_stall=m.Staleness())
eager = m.CoarseCorrection()
CONFIGS = {
    # fixed rungs (ZZ_TB=1 turns the Schwarz rung into term blocks)
    "F_schwarz": P.Schwarz(),
    "F_coarse": P.Schwarz(coarse=eager),
    # climbs
    "OL": P.Adaptive(),  # diag -> schwarz
    "A": P.Adaptive(coarse=eager),  # diag -> coarse
    "B": P.Adaptive(coarse=lazy),  # diag -> schwarz -> coarse
}
if config == "Bsw":
    st = m.Staleness(
        window=int(os.environ["ZZ_W"]), threshold=float(os.environ["ZZ_T"])
    )
    CONFIGS["Bsw"] = P.Adaptive(stall=st, coarse=m.CoarseCorrection(on_stall=st))
cfg = CONFIGS[config]
t0 = time.perf_counter()
solver = w.Solver(effects, weights, cfg)
t1 = time.perf_counter()
r = solver.solve(y, w.LsmrOptions(tol=1e-8, maxiter=20_000))
t2 = time.perf_counter()
print(
    json.dumps(
        {
            "construct": t1 - t0,
            "solve": t2 - t1,
            "total": t2 - t0,
            "build_in_solve": r.time_setup,
            "iterations": r.iterations,
            "converged": r.converged,
            "residual": r.residual,
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        }
    )
)
