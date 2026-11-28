# ruff: noqa
import json, sys, statistics as st

rows = [json.loads(l) for l in open(sys.argv[1])]
g = {}
for r in rows:
    if "error" in r:
        print("ERR", r)
        continue
    g.setdefault(
        (
            r["panel"],
            r.get("build", "") + ":" + r["config"],
            r["knobs"],
            r.get("batch", 0),
        ),
        [],
    ).append(r)


def med(v):
    return st.median(v), min(v), max(v)


print(
    f"{'panel':42s} {'cfg':22s} n  it      construct s        solve s (min,max)       build_in  setupMB endMB  B/obs(end-loaded) coarse cert shas"
)
for (p, c, k, b), rs in sorted(g.items()):
    n = rs[0]["n_obs"]
    it = rs[0]["iterations"]
    co = med([r["construct"] for r in rs])
    so = med([r["solve"] for r in rs])
    bi = st.median([sum(r["build_in_solve"]) for r in rs])
    su = st.median([r["rss_setup_mb"] for r in rs])
    en = med([r["rss_end_mb"] for r in rs])
    lo = st.median([r["rss_loaded_mb"] for r in rs])
    shas = {r["x_sha"] for r in rs}
    its = {tuple(r["iterations"]) for r in rs}
    print(
        f"{p[4:]:42s} {(c + ' ' + k)[:22]:22s} {len(rs)} {str(it) if len(it) < 3 else sum(it) / len(it):7} {co[0]:6.2f} {so[0]:7.2f} ({so[1]:.2f},{so[2]:.2f}) {bi:6.2f} {su:7.0f} {en[0]:6.0f} [{en[1]:.0f},{en[2]:.0f}] {(en[0] - lo) * 2**20 / n:6.0f} {rs[0]['coarse']} {all(all(r['certified'] or []) for r in rs)} {len(shas)}/{len(its)}"
    )
