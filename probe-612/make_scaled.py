# ruff: noqa
"""Stalled AKM cells at several sizes, same draws as probe-607's make_panels: npz with codes_i, slopes_i_j, y."""

import sys
from pathlib import Path
import numpy as np

port, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(port))
from benchmarks.suites.akm_slopes import _LADDER, _SPECS, _mover_panel
from benchmarks._problems import _make_response

specs = dict(_SPECS)
jobs = [(d, s, n) for d, s, n in (x.split(":") for x in sys.argv[3:])]
for design, spec, n in jobs:
    n = int(float(n))
    name = f"akm_{design.replace('=', '')}_{spec}_{n // 1000}k"
    if (out / f"{name}.npz").exists():
        continue
    panel = _mover_panel(n, _LADDER[design], 0)
    rng = np.random.default_rng(0)
    y = _make_response(panel.codes, panel.n_levels, rng)
    worker, firm = specs[spec](panel, rng)
    for factor, slopes in ((0, worker), (1, firm)):
        for z in slopes or ():
            y += (
                z
                * (0.5 * rng.standard_normal(panel.n_levels[factor]))[
                    panel.codes[factor]
                ]
            )
    arrays = {"y": y}
    for i, (c, sl) in enumerate(zip(panel.codes, [worker, firm, None])):
        arrays[f"codes_{i}"] = c
        for j, z in enumerate(sl or []):
            arrays[f"slopes_{i}_{j}"] = np.asarray(z, dtype=np.float64)
    np.savez(out / f"{name}.npz", **arrays)
    print(name, len(y), panel.n_levels, flush=True)
