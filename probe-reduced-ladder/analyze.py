import json
import math
import sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1])]
cell = defaultdict(list)
for r in rows:
    cell[(r["panel"], r["variant"])].append(r)

HARD = ("akm_mobility0.05", "akm_mobility0.001", "akm_sorting1e4")
CLIMBS = ["A", "B", "A+TB", "B+TB", "A_tbc", "B_tbc"]
FIXED = ["F_schwarz", "F_tb", "F_coarse", "F_tbcoarse"]


def stat(rs):
    ok = [r for r in rs if "error" not in r]
    if not ok:
        return None
    return {
        "t": min(r["total"] for r in ok),
        "it": sorted(r["iterations"] for r in ok)[len(ok) // 2],
        "conv": all(r["converged"] for r in ok),
        "rss": max(r["maxrss_mb"] for r in ok),
        "build": min(r["build_in_solve"] + r["construct"] for r in ok),
        "n": len(ok),
        "err": len(rs) - len(ok),
    }


panels = sorted({p for p, _ in cell})
variants = ["OL"] + FIXED + CLIMBS
S = {(p, v): stat(cell[(p, v)]) for p in panels for v in variants if (p, v) in cell}

print(f"{'panel':38}" + "".join(f"{v:>14}" for v in variants))
for p in panels:
    line = f"{p:38}"
    for v in variants:
        s = S.get((p, v))
        line += (
            f"{'':>14}"
            if s is None
            else f"{s['t']:7.2f}s/{s['it']:<4}{'' if s['conv'] else '!'}{'E' if s['err'] else ''}".rjust(
                14
            )
        )
    print(line)

print("\nRSS MB (max over rounds)")
for p in panels:
    print(
        f"{p:38}"
        + "".join(
            f"{S[(p, v)]['rss']:14.0f}" if S.get((p, v)) else f"{'':>14}"
            for v in variants
        )
    )


def geo(vals):
    vals = [v for v in vals if v]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


hard = [p for p in panels if p.startswith(HARD)]
easy = [p for p in panels if not p.startswith(HARD)]
print("\nHard cells: geo-mean time ratio to A (diag -> coarse)")
for v in CLIMBS:
    print(
        f"  {v:8}",
        f"{geo([S[(p, v)]['t'] / S[(p, 'A')]['t'] for p in hard if S.get((p, v)) and S.get((p, 'A'))]):.3f}",
        "  max",
        f"{max((S[(p, v)]['t'] / S[(p, 'A')]['t'] for p in hard if S.get((p, v)) and S.get((p, 'A'))), default=float('nan')):.3f}",
    )
print("\nEasy cells: time ratio to OL (diag -> schwarz); worst cell")
for v in [v for v in CLIMBS if v != "B_tbc"]:
    rat = [
        (S[(p, v)]["t"] / S[(p, "OL")]["t"], p)
        for p in easy
        if S.get((p, v)) and S.get((p, "OL"))
    ]
    if rat:
        print(
            f"  {v:8} geo {geo([r for r, _ in rat]):.3f}  worst {max(rat)[0]:.3f} {max(rat)[1]}"
        )
print("\nFixed rungs: TB / Schwarz (time, iterations); TB+coarse / coarse")
for p in panels:
    a, b, c, d = (S.get((p, v)) for v in FIXED)
    parts = []
    if a and b:
        parts.append(
            f"tb/schwarz t {b['t'] / a['t']:.2f} it {b['it']}/{a['it']} build {b['build']:.2f}/{a['build']:.2f}"
        )
    if c and d:
        parts.append(
            f"tbc/coarse t {d['t'] / c['t']:.2f} it {d['it']}/{c['it']} build {d['build']:.2f}/{c['build']:.2f}"
        )
    print(f"  {p:38}", " | ".join(parts))
