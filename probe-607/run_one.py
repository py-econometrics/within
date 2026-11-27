"""One solve of one panel under one config on 8 threads; prints a JSON line."""

import hashlib
import json
import os
import resource
import sys
import tempfile
import time
import numpy as np
import within as w
import within._within as m


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS: bytes


panel, config = sys.argv[1], sys.argv[2]
d = np.load(panel)
n_terms = sum(1 for k in d.files if k.startswith("codes_"))
effects = []
for i in range(n_terms):
    slopes = [d[k] for k in sorted(d.files) if k.startswith(f"slopes_{i}_")]
    effects.append(w.Effect(d[f"codes_{i}"], True, slopes or None))
weights = d["weights"] if "weights" in d.files else None
y = d["y"]
rss_loaded = rss_mb()

P = w.PreconditionerConfig
st = m.Staleness(window=3, threshold=0.5)
if config == "OL":  # main's default ladder: diagonal -> Schwarz (4, 0.7)
    cfg = P.Adaptive()
elif config == "F_diag":  # fixed diagonal (BlockDiagonal under PROBE_BOTTOM=block)
    cfg = P.Diagonal()
elif config == "F_schwarz":  # fixed one-level Schwarz, built in construct
    cfg = P.Additive() if hasattr(P, "Additive") else P.Schwarz()
elif (
    config == "B"
):  # decided climb minus BlockDiagonal: diagonal -> Schwarz -> coarse at (3, 0.5)
    cfg = P.Adaptive(stall=st, coarse=m.CoarseCorrection(on_stall=st))
elif config.startswith("P_"):  # prototype knobs come through WITHIN_PROTO_* env vars
    cfg = P.Adaptive()
elif config == "F_coarse":  # fixed two-level, built in construct
    cfg = P.Schwarz(coarse=m.CoarseCorrection())
# Capture the Rust side's stderr: the solver's own audit value and the coarse rung's fate.

os.environ["PROBE_TRACE"] = "1"
_errf = tempfile.TemporaryFile(mode="w+b")
_saved = os.dup(2)
os.dup2(_errf.fileno(), 2)
t0 = time.perf_counter()
try:
    _t = int(os.environ.get("PROBE_THREADS", "8"))
    solver = w.Solver(effects, weights, cfg, threads=_t)
    kw = {"threads": _t}
except (
    TypeError
):  # feature/reduced-solve: no threads argument; RAYON_NUM_THREADS=8 bounds the pool
    solver = w.Solver(effects, weights, cfg)
    kw = {}
t1 = time.perf_counter()
rss_setup = rss_mb()
r = solver.solve(
    y,
    w.LsmrOptions(
        tol=float(os.environ.get("PROBE_TOL", "1e-8")),
        maxiter=int(os.environ.get("PROBE_MAXITER", "20000")),
    ),
    **kw,
)
t2 = time.perf_counter()


def rung_of(solver):
    pc = solver.preconditioner
    if pc is not None and hasattr(pc, "rung"):  # feature/reduced-solve
        return repr(pc.rung)
    return "Schwarz" if solver.has_escalated else "Diagonal"


os.dup2(_saved, 2)
_errf.seek(0)
_err = _errf.read().decode(errors="replace")
sys.stderr.write(_err)
_cert = [ln for ln in _err.splitlines() if ln.startswith("[certify]")]
audit_backward = (
    float(_cert[-1].split("backward_error=")[1].split()[0]) if _cert else None
)
_audits = [ln for ln in _err.splitlines() if ln.startswith("[audit] it=")]
audit_certified = ("Certified" in _audits[-1]) if _audits else None
_coarse = [ln for ln in _err.splitlines() if ln.startswith("[probe-coarse]")]
coarse = _coarse[-1].split()[1] if _coarse else None
measurements = sum(ln.startswith("[measure]") for ln in _err.splitlines())
# Certification measured outside the solver, on the full weighted system:
# g = D^T W r per column, rel = |g| / |D^T W y|, be = |g| / (sqrt(max diag G) * |W^1/2 r|).
wt = weights if weights is not None else np.ones_like(y)
res = np.asarray(r.demeaned)
g2 = gy2 = 0.0
dmax = 0.0
for e_i in range(n_terms):
    c = d[f"codes_{e_i}"]
    cols = [np.ones_like(y)] + [
        d[k] for k in sorted(d.files) if k.startswith(f"slopes_{e_i}_")
    ]
    for z in cols:
        g2 += float(np.sum(np.bincount(c, wt * z * res) ** 2))
        gy2 += float(np.sum(np.bincount(c, wt * z * y) ** 2))
        dmax = max(dmax, float(np.bincount(c, wt * z * z).max()))
grad_rel = (g2 / gy2) ** 0.5
backward = g2**0.5 / (dmax**0.5 * float(np.sum(wt * res * res)) ** 0.5)
rung = config if config.startswith("F_") else rung_of(solver)
print(
    json.dumps(
        {
            "panel": panel.rsplit("/", 1)[-1][:-4],
            "n_obs": len(y),
            "construct": t1 - t0,
            "solve": t2 - t1,
            "build_in_solve": r.time_setup,
            "iterations": r.iterations,
            "converged": r.converged,
            "residual": r.residual,
            "rung": rung,
            "rss_loaded_mb": rss_loaded,
            "rss_setup_mb": rss_setup,
            "rss_end_mb": rss_mb(),
            "audit_backward": audit_backward,
            "audit_certified": audit_certified,
            "coarse": coarse,
            "measurements": measurements,
            "grad_rel": grad_rel,
            "backward": backward,
            "x_sha": hashlib.sha256(np.ascontiguousarray(r.x).tobytes()).hexdigest()[
                :16
            ],
        }
    )
)
