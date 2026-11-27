"""Write every benchmark panel as an npz: codes_i, slopes_i_j, y, weights."""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
import pyarrow.compute as pc

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
CACHE = Path.home() / ".cache/within-bench"


def codes_of(a):
    _, inv = np.unique(np.asarray(a), return_inverse=True)
    return inv.astype(np.uint32)


def save(name, codes, slopes, y, weights=None):
    arrays = {"y": np.asarray(y, dtype=np.float64)}
    for i, c in enumerate(codes):
        arrays[f"codes_{i}"] = codes_of(c)
        for j, z in enumerate(slopes[i] or []):
            arrays[f"slopes_{i}_{j}"] = np.asarray(z, dtype=np.float64)
    if weights is not None:
        arrays["weights"] = np.asarray(weights, dtype=np.float64)
    np.savez(OUT / f"{name}.npz", **arrays)
    print(
        name,
        len(arrays["y"]),
        [int(arrays[f"codes_{i}"].max()) + 1 for i in range(len(codes))],
        flush=True,
    )


def synthetic():
    from benchmarks.suites.akm_slopes import _LADDER, _SPECS, _mover_panel
    from benchmarks._problems import _make_response, fixest_dgp, random_kfe

    specs = dict(_SPECS)
    for design in ["mobility=1.0", "mobility=0.05", "mobility=0.001", "sorting=1e4"]:
        panel = _mover_panel(1_000_000, _LADDER[design], 0)
        for spec in ["intercepts", "near_shared_year", "experience_tenure"]:
            # Same draws as akm_slopes._slope_case.
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
            save(
                f"akm_{design.replace('=', '')}_{spec}",
                panel.codes,
                [worker, firm, None],
                y,
            )
    # 10M-row low-mobility panel for the memory side of the goal
    panel = _mover_panel(10_000_000, _LADDER["mobility=0.001"], 0)
    rng = np.random.default_rng(0)
    y = _make_response(panel.codes, panel.n_levels, rng)
    save("akm10m_mobility0.001_intercepts", panel.codes, [None, None, None], y)
    cats, _, y = fixest_dgp(1_000_000, "difficult", n_fe=3, seed=0)
    save("fixest3_difficult", cats, [None] * 3, y)
    cats, _, y = random_kfe(5, [20_000, 20_000, 5_000, 1_000, 100], 1_000_000, seed=0)
    save("random5fe", cats, [None] * 5, y)


def real():
    t = pq.read_table(
        CACHE / "baci/baci_hs22_2023.parquet",
        columns=["exporter", "importer", "product", "lv"],
    )
    save(
        "baci",
        [t[c].to_numpy() for c in ["exporter", "importer", "product"]],
        [None] * 3,
        t["lv"].to_numpy(),
    )

    t = pq.read_table(
        CACHE / "nyc_taxi/nyc_taxi_2012q1.parquet",
        columns=["puloc", "doloc", "pdate", "tip_amount"],
    )
    save(
        "taxi",
        [t[c].to_numpy() for c in ["puloc", "doloc", "pdate"]],
        [None] * 3,
        t["tip_amount"].to_numpy(),
    )

    parts = [
        pcsv.read_csv(
            CACHE / f"hmda/us_cells_{y}.csv",
            convert_options=pcsv.ConvertOptions(
                column_types={"county_code": "string", "lei": "string"}
            ),
        )
        for y in (2018, 2019, 2020)
    ]
    year = np.concatenate([p["activity_year"].to_numpy() for p in parts]).astype(
        np.float64
    )
    lei = np.concatenate([p["lei"].to_numpy(zero_copy_only=False) for p in parts])
    county = np.concatenate(
        [p["county_code"].to_numpy(zero_copy_only=False) for p in parts]
    )
    count = np.concatenate([p["count"].to_numpy() for p in parts]).astype(np.float64)
    rate = np.concatenate([p["sum_rate"].to_numpy() for p in parts]) / count
    t = year - 2019.0
    keep = ~(np.equal(county, None) | np.equal(lei, None))
    save(
        "hmda_us",
        [lei[keep], county[keep], year[keep]],
        [[t[keep]], [t[keep]], None],
        rate[keep],
        count[keep],
    )

    g = pcsv.read_csv(
        CACHE / "gravity/Gravity_V202211.csv",
        convert_options=pcsv.ConvertOptions(
            include_columns=[
                "year",
                "iso3_o",
                "iso3_d",
                "country_exists_o",
                "country_exists_d",
            ]
        ),
    )
    ok = pc.and_(pc.equal(g["country_exists_o"], 1), pc.equal(g["country_exists_d"], 1))
    g = g.filter(ok)
    yr = g["year"].to_numpy()
    o = g["iso3_o"].to_numpy(zero_copy_only=False).astype(str)
    d = g["iso3_d"].to_numpy(zero_copy_only=False).astype(str)
    rng = np.random.default_rng(0)
    save(
        "gravity",
        [
            np.char.add(o, yr.astype(str)),
            np.char.add(d, yr.astype(str)),
            np.char.add(o, d),
        ],
        [None] * 3,
        rng.standard_normal(len(yr)),
    )


if __name__ == "__main__":
    {"synthetic": synthetic, "real": real}[sys.argv[2]]()
