# ruff: noqa
"""One solve (or batch) of one panel; prints a JSON line. Configs: OL (main's ladder; PROBE_COARSE=climb
adds #607's rigid rung), F_schwarz (fixed one-level Schwarz). PROBE_BATCH=k solves k responses at once.
PROBE_RITZ=path writes the first LSMR pass's Ritz values of the preconditioned normal operator."""

import hashlib, json, os, resource, sys, tempfile, time
import numpy as np
import within as w


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


panel, config = sys.argv[1], sys.argv[2]
d = np.load(panel)
n_terms = sum(1 for k in d.files if k.startswith("codes_"))
effects = []
for i in range(n_terms):
    slopes = [d[k] for k in sorted(d.files) if k.startswith(f"slopes_{i}_")]
    effects.append(w.Effect(d[f"codes_{i}"], True, slopes or None))
y = d["y"]
k = int(os.environ.get("PROBE_BATCH", "0"))
if k:
    rng = np.random.default_rng(1)
    Y = np.column_stack([y] + [y + rng.standard_normal(len(y)) for _ in range(k - 1)])
rss_loaded = rss_mb()
P = w.PreconditionerConfig
cfg = {
    "OL": lambda: P.Adaptive(),
    "F_schwarz": lambda: P.Additive() if hasattr(P, "Additive") else P.Schwarz(),
}[config]()
ritz_path = os.environ.get("PROBE_RITZ")
os.environ["PROBE_TRACE"] = "1"
errf = tempfile.TemporaryFile(mode="w+b")
saved = os.dup(2)
os.dup2(errf.fileno(), 2)
threads = int(os.environ.get("PROBE_THREADS", "8"))
opts = w.LsmrOptions(
    tol=float(os.environ.get("PROBE_TOL", "1e-8")),
    maxiter=int(os.environ.get("PROBE_MAXITER", "20000")),
)
t0 = time.perf_counter()
solver = w.Solver(effects, None, cfg, threads=threads)
t1 = time.perf_counter()
rss_setup = rss_mb()
if k:
    rb = solver.solve_batch(Y, opts, threads=threads)
    from types import SimpleNamespace as NS

    its, conv, res, X = (
        list(rb.iterations),
        list(rb.converged),
        list(np.atleast_1d(rb.residual)),
        np.asarray(rb.x),
    )
    ts = np.atleast_1d(rb.time_setup)
    results = [
        NS(
            iterations=int(its[j]),
            converged=bool(conv[j]),
            residual=float(res[j]),
            x=X[:, j] if X.shape[0] != k else X[j],
            time_setup=float(ts[j] if len(ts) > 1 else (ts[0] if j == 0 else 0.0)),
        )
        for j in range(k)
    ]
else:
    results = [solver.solve(y, opts, threads=threads)]
t2 = time.perf_counter()
os.dup2(saved, 2)
errf.seek(0)
err = errf.read().decode(errors="replace")
lines = err.splitlines()
coarse = [ln for ln in lines if ln.startswith("[probe-coarse]")]
cert = [ln for ln in lines if ln.startswith("[audit] it=")]
rec = {
    "panel": os.path.basename(panel)[:-4],
    "n_obs": len(y),
    "config": config,
    "batch": k,
    "construct": t1 - t0,
    "solve": t2 - t1,
    "iterations": [r.iterations for r in results],
    "converged": [bool(r.converged) for r in results],
    "residual": [r.residual for r in results],
    "build_in_solve": [r.time_setup for r in results],
    "probe": [
        ln
        for ln in lines
        if ln.startswith("[probe-space]") or ln.startswith("[probe-coarse]")
    ],
    "rung": "Schwarz" if solver.has_escalated else "Diagonal",
    "coarse": coarse[-1].split()[1] if coarse else None,
    "certified": ["Certified" in c for c in cert][-len(results) :] if cert else None,
    "rss_loaded_mb": rss_loaded,
    "rss_setup_mb": rss_setup,
    "rss_end_mb": rss_mb(),
    "x_sha": hashlib.sha256(
        b"".join(np.ascontiguousarray(r.x).tobytes() for r in results)
    ).hexdigest()[:16],
}
if ritz_path:
    passes, cur = [], None
    for ln in lines:
        if ln.startswith("[ritz]"):
            kv = dict(t.split("=") for t in ln.split()[1:])
            if "pass" in kv:
                cur = {"a": [float(kv["a"])], "b": []}
                passes.append(cur)
            else:
                cur["b"].append(float(kv["b"]))
                cur["a"].append(float(kv["a"]))
    p = passes[0]
    m = len(p["b"])  # alphas a1..a_{m+1}, betas b2..b_{m+1}
    B = np.zeros((m + 1, m))
    for i in range(m):
        B[i, i] = p["a"][i]
        B[i + 1, i] = p["b"][i]
    sv = np.linalg.svd(B, compute_uv=False)
    np.savez(ritz_path, lam=np.sort(sv**2), passes=len(passes), steps=m)
    lam = np.sort(sv**2)
    top = lam[-1]
    rec["ritz"] = {
        "steps": m,
        "lam_max": top,
        "lam_min": lam[0],
        "below": {f"1e-{j}": int(np.sum(lam < top * 10**-j)) for j in (1, 2, 3, 4, 6)},
    }
print(json.dumps(rec))
