import json
import math
import re
import sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1]) if "error" not in line]
HARD = ("akm_mobility0.05", "akm_mobility0.001", "akm_sorting1e4")


def rung(r):
    ats = [
        int(m.group(1))
        for line in r["trace"]
        for m in [re.search(r"rung=Some\((\d)\)", line)]
        if m
    ]
    return max(ats) if ats else 0


by = defaultdict(dict)
for r in rows:
    by[(r["window"], r["threshold"])][r["panel"]] = (
        r["iterations"],
        rung(r),
        r["converged"],
        r["total"],
    )

base = by[(4, 0.7)]
panels = sorted(base)
hard = [p for p in panels if p.startswith(HARD)]
easy = [p for p in panels if not p.startswith(HARD)]


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


print(
    f"{'w':>2} {'t':>4} {'hard it geo':>11} {'vs 4/0.7':>8} {'easy it geo':>11} {'easy more it':>12} {'easy extra rung':>15} {'unconv':>6}"
)
for (w, t), res in sorted(by.items()):
    if set(res) != set(panels):
        continue
    hg = geo([res[p][0] for p in hard])
    eg = geo([res[p][0] for p in easy])
    more = [p for p in easy if res[p][0] > base[p][0]]
    extra = [p for p in easy if res[p][1] > base[p][1]]
    unconv = sum(not res[p][2] for p in panels)
    print(
        f"{w:>2} {t:>4} {hg:11.1f} {hg / geo([base[p][0] for p in hard]):8.3f} {eg:11.1f} {len(more):12} {len(extra):15} {unconv:6}"
    )

print("\nper panel iterations/rung at 4/0.7:")
for p in panels:
    print(f"  {p:38} {base[p][0]:5} rung {base[p][1]}")
