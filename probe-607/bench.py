"""Interleaved rotated rounds over (build, config) variants; 5 rounds below 5M rows, 3 above.
usage: bench.py OUT VARIANTS [PANEL_GLOB] ; VARIANTS = build/config,build/config,..."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

S = Path(__file__).parent
# VARIANTS = build/config[/ENV=V ENV=V];... ; env knobs travel with the variant
out_name = sys.argv[1]
variants = [tuple((v.split("/", 2) + [""])[:3]) for v in sys.argv[2].split(";")]
PANELS = sorted((S / "panels").glob(sys.argv[3] if len(sys.argv) > 3 else "*.npz"))
rows = {p: len(np.load(p, mmap_mode="r")["y"]) for p in PANELS}
rounds = {p: 5 if rows[p] < 5_000_000 else 3 for p in PANELS}
out = open(S / f"{out_name}.jsonl", "a")
log = open(S / f"{out_name}.log", "a", buffering=1)
total = sum(rounds.values()) * len(variants)
done = 0
t0 = time.time()
for rnd in range(max(rounds.values())):
    for i, p in enumerate(PANELS):
        if rnd >= rounds[p]:
            continue
        k = (rnd + i) % len(variants)
        for build, cfg, knobs in variants[k:] + variants[:k]:
            env = {
                **os.environ,
                "RAYON_NUM_THREADS": "8",
                **dict(kv.split("=", 1) for kv in knobs.split()),
            }
            t = time.time()
            proc = subprocess.run(
                [
                    str(S / f"venv-{build}/bin/python"),
                    "-I",
                    str(S / "run_one.py"),
                    str(p),
                    cfg,
                ],
                capture_output=True,
                text=True,
                timeout=3600,
                env=env,
            )
            line = (
                proc.stdout.strip().splitlines()[-1]
                if proc.returncode == 0 and proc.stdout.strip()
                else None
            )
            rec = (
                json.loads(line)
                if line
                else {"panel": p.stem, "error": proc.stderr.strip()[-400:]}
            )
            rec.update(build=build, config=cfg, knobs=knobs, round=rnd)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done += 1
            el = time.time() - t0
            log.write(
                f"{time.strftime('%T')} run {done}/{total} r{rnd} {p.stem} {build}/{cfg}/{knobs} "
                f"{time.time() - t:.1f}s it={rec.get('iterations')} eta={el / done * (total - done) / 60:.0f}min\n"
            )
