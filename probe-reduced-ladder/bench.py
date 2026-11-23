"""Rotated rounds over (panel, variant); appends JSON lines to results.jsonl."""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
W = os.environ.get("WITHIN_WORKTREE", ".")

# variant -> (config, env)
VARIANTS = {
    "OL": ("OL", {}),
    "F_schwarz": ("F_schwarz", {}),
    "F_tb": ("F_schwarz", {"ZZ_TB": "1"}),
    "F_coarse": ("F_coarse", {}),
    "F_tbcoarse": ("F_coarse", {"ZZ_TB": "1"}),
    "A": ("A", {}),  # diag -> coarse
    "B": ("B", {}),  # diag -> schwarz -> coarse
    "A+TB": ("A", {"ZZ_TB_RUNG": "1"}),  # diag -> tb -> coarse
    "B+TB": ("B", {"ZZ_TB_RUNG": "1"}),  # diag -> tb -> schwarz -> coarse
    "A_tbc": ("A", {"ZZ_TB": "1"}),  # diag -> tb+coarse
    "B_tbc": ("B", {"ZZ_TB": "1"}),  # diag -> tb -> tb+coarse
}
HARD = ["akm_mobility0.05", "akm_mobility0.001", "akm_sorting1e4"]
EASY_VARIANTS = [
    "OL",
    "F_schwarz",
    "F_tb",
    "F_coarse",
    "F_tbcoarse",
    "A",
    "B",
    "A+TB",
    "B+TB",
    "A_tbc",
]
HARD_VARIANTS = ["F_coarse", "F_tbcoarse", "A", "B", "A+TB", "B+TB", "A_tbc", "B_tbc"]


def plan(panels):
    for p in panels:
        hard = any(p.startswith(h) for h in HARD)
        for v in HARD_VARIANTS if hard else EASY_VARIANTS:
            yield p, v


def main():
    rounds = int(sys.argv[1])
    panels = sys.argv[2].split(",")
    out = HERE / sys.argv[3]
    jobs = list(plan(panels))
    for rnd in range(rounds):
        # rotate the order each round
        k = (rnd * len(jobs)) // max(rounds, 1)
        for p, v in jobs[k:] + jobs[:k]:
            config, env = VARIANTS[v]
            e = dict(
                os.environ,
                RAYON_NUM_THREADS="8",
                ZZ_TRACE="1",
                PYTHONPATH=f"{W}/python",
                **env,
            )
            proc = subprocess.run(
                [
                    PY,
                    str(HERE / "run_one.py"),
                    str(HERE / "panels" / f"{p}.npz"),
                    config,
                ],
                env=e,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            rec = {"round": rnd, "panel": p, "variant": v}
            try:
                rec.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except Exception:
                rec["error"] = (proc.stderr or proc.stdout)[-500:]
            rec["trace"] = [
                line for line in proc.stderr.splitlines() if line.startswith("ZZ ")
            ]
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                rnd,
                p,
                v,
                rec.get("iterations"),
                round(rec.get("total", -1), 3),
                rec.get("error", "")[:80],
                flush=True,
            )


if __name__ == "__main__":
    main()
