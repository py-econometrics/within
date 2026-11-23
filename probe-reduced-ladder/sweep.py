"""Iteration-only stall sweep for path B; one run per (panel, window, threshold)."""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
W = os.environ.get("WITHIN_WORKTREE", ".")
panels = sys.argv[1].split(",")
out = HERE / "sweep.jsonl"
WINS = tuple(int(x) for x in sys.argv[2].split(","))
THRS = tuple(float(x) for x in sys.argv[3].split(","))
for p in panels:
    for win in WINS:
        for thr in THRS:
            e = dict(
                os.environ,
                RAYON_NUM_THREADS="8",
                ZZ_TRACE="1",
                PYTHONPATH=f"{W}/python",
                ZZ_W=str(win),
                ZZ_T=str(thr),
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_one.py"),
                    str(HERE / "panels" / f"{p}.npz"),
                    "Bsw",
                ],
                env=e,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            rec = {"panel": p, "window": win, "threshold": thr}
            try:
                rec.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except Exception:
                rec["error"] = (proc.stderr or proc.stdout)[-400:]
            rec["trace"] = [
                line for line in proc.stderr.splitlines() if line.startswith("ZZ ")
            ]
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                p,
                win,
                thr,
                rec.get("iterations"),
                rec["trace"][-1:],
                rec.get("error", "")[:80],
                flush=True,
            )
