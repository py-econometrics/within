# ruff: noqa
"""Rounds over (panel, variant), order rotated each round; one JSON line per run, a progress log beside.
usage: bench.py OUT ROUNDS_SMALL ROUNDS_LARGE 'venv:config|ENV=V ENV=V;...' panel...  (venv: base or cand)"""

import json, os, subprocess, sys, time
from pathlib import Path

S = Path(__file__).resolve().parent.parent
out_name, r_small, r_large = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
variants = [
    tuple((v.split("|") + [""])[:2]) for v in sys.argv[4].split(";")
]  # ("base:OL", knobs)
panels = [Path(p) for p in sys.argv[5:]]
large = lambda p: p.stat().st_size > 100e6
rounds = {p: r_large if large(p) else r_small for p in panels}
out = open(S / f"{out_name}.jsonl", "a")
log = open(S / f"{out_name}.log", "a", buffering=1)
total = sum(rounds.values()) * len(variants)
done = 0
t0 = time.time()
for rnd in range(max(rounds.values())):
    for i, p in enumerate(panels):
        if rnd >= rounds[p]:
            continue
        k = (rnd + i) % len(variants)
        for cfg, knobs in variants[k:] + variants[:k]:
            env = {
                **os.environ,
                "RAYON_NUM_THREADS": "8",
                **dict(kv.split("=", 1) for kv in knobs.split()),
            }
            t = time.time()
            proc = subprocess.run(
                [
                    str(S / f"venv-{cfg.split(':')[0]}/bin/python"),
                    "-I",
                    str(S / "tools/run_one.py"),
                    str(p),
                    cfg.split(":")[1],
                ],
                capture_output=True,
                text=True,
                timeout=7200,
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
            rec.update(knobs=knobs, round=rnd, build=cfg.split(":")[0])
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done += 1
            el = time.time() - t0
            log.write(
                f"{time.strftime('%T')} run {done}/{total} r{rnd} {p.stem} {cfg}|{knobs} {time.time() - t:.1f}s "
                f"it={rec.get('iterations')} eta={el / done * (total - done) / 60:.0f}min\n"
            )
