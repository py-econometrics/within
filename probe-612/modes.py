# ruff: noqa
"""Assemble the reduced S (worker eliminated, kept: firm [1, slope], year [1]) of an AKM panel and
look at its near-null modes under Jacobi scaling: how much of each lies in candidate coarse spaces."""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

d = np.load(sys.argv[1])
nev = int(sys.argv[2]) if len(sys.argv) > 2 else 40
wk, fm, yr = (d[f"codes_{i}"].astype(np.int64) for i in range(3))
xw = d["slopes_0_0"]
xf = d["slopes_1_0"]
n = len(wk)
nw, nf, ny = wk.max() + 1, fm.max() + 1, yr.max() + 1
rows = np.arange(n)
# the solve basis: each firm's slope centered on the firm (orthogonal to its intercept)
xf = xf - (np.bincount(fm, xf, nf) / np.maximum(np.bincount(fm, minlength=nf), 1))[fm]
# kept columns: firm intercept [0,nf), firm slope [nf,2nf), year [2nf, 2nf+ny)
K = sp.csr_matrix(
    (
        np.concatenate([np.ones(n), xf, np.ones(n)]),
        (np.concatenate([rows] * 3), np.concatenate([fm, nf + fm, 2 * nf + yr])),
    ),
    shape=(n, 2 * nf + ny),
)
# per worker orthonormal basis of [1, xw] on its rows
order = np.argsort(wk, kind="stable")
starts = np.searchsorted(wk[order], np.arange(nw + 1))
q_rows, q_cols, q_vals = [], [], []
for g in range(nw):
    r = order[starts[g] : starts[g + 1]]
    E = np.column_stack([np.ones(len(r)), xw[r]])
    Q, R = np.linalg.qr(E)
    keep = np.abs(np.diag(R)) > 1e-10 * np.abs(R[0, 0])
    for j in np.nonzero(keep)[0]:
        q_rows += [2 * g + j] * len(r)
        q_cols += list(r)
        q_vals += list(Q[:, j])
Qt = sp.csr_matrix((q_vals, (q_rows, q_cols)), shape=(2 * nw, n))
C = Qt @ K
Sm = (K.T @ K - C.T @ C).tocsc()
dg = Sm.diagonal()
kn = (K.T @ K).diagonal()
live = dg > 1e-10 * kn
# a firm level whose own Gram of [1, x] is rank-deficient keeps only its intercept, as the solve basis does
cnt = np.bincount(fm, minlength=nf)
sx = np.bincount(fm, xf, nf)
sxx = np.bincount(fm, xf * xf, nf)
var = sxx - sx * sx / np.maximum(cnt, 1)
live[nf : 2 * nf] &= sxx > 1e-20 * cnt
Dm = sp.diags(1 / np.sqrt(dg[live]))
A = (Dm @ Sm[live][:, live] @ Dm).tocsc()
m = A.shape[0]
# exact nulls: firm intercept const (+ year const), etc. Shift to see past them.
vals, vecs = sla.eigsh(A + 1e-12 * sp.identity(m), k=nev, sigma=0, which="LM")
o = np.argsort(vals)
vals, vecs = vals[o] - 1e-12, vecs[:, o]
lam_max = sla.eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
idx = np.nonzero(live)[0]
blk = lambda v: (
    np.linalg.norm(v[idx < nf]),
    np.linalg.norm(v[(idx >= nf) & (idx < 2 * nf)]),
    np.linalg.norm(v[idx >= 2 * nf]),
)
# candidate: global gauge per channel (Jacobi-scaled): constant firm slope, constant firm intercept, constant year
G = np.zeros((m, 3))
G[idx < nf, 0] = np.sqrt(dg[idx[idx < nf]])
G[(idx >= nf) & (idx < 2 * nf), 1] = np.sqrt(dg[idx[(idx >= nf) & (idx < 2 * nf)]])
G[idx >= 2 * nf, 2] = np.sqrt(dg[idx[idx >= 2 * nf]])
Gq, _ = np.linalg.qr(G)
print(f"n={n} kept={m} dropped={(~live).sum()} lam_max={lam_max:.3f}")
for i in range(nev):
    v = vecs[:, i]
    fi, fs, y = blk(v)
    print(
        f"{i:3d} lam={vals[i]:.3e} rel={vals[i] / lam_max:.2e} |firm_int|={fi:.2f} |firm_slope|={fs:.2f} |year|={y:.2f} in_global_gauges={np.linalg.norm(Gq.T @ v):.3f} support={int((np.abs(v) > 0.01 * np.abs(v).max()).sum())}"
    )
np.savez(
    sys.argv[1].replace(".npz", "_modes.npz"),
    vals=vals,
    vecs=vecs,
    idx=idx,
    lam_max=lam_max,
)
