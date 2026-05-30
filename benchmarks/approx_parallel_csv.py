"""Probe one-shot approximate Schwarz residualization on unpacked CSV benchmarks.

Usage:

    python benchmarks/approx_parallel_csv.py --datasets credit2 soccer synthetic-zigzag

The script expects ``benchmarks/all-csv`` to contain the CSV files and metadata
JSONs from ``all-csv.zip``. It compares the new one-shot approximate solve to
the corrected LSMR solve using the same cached preconditioner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import within


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readline().strip().split(",")


def _factorize(values: np.ndarray) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.uint32, copy=False), int(inverse.max(initial=-1) + 1)


def _factor_columns(metadata: dict, mode: str) -> list[str]:
    graph_cols = [metadata["graph_id1"], metadata["graph_id2"]]
    if mode == "graph":
        return graph_cols
    extras = metadata.get("additional_identifier_columns", [])
    return [*graph_cols, *extras]


def load_problem(data_dir: Path, slug: str, factor_mode: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    metadata_path = data_dir / "metadata" / f"{slug}.json"
    csv_path = data_dir / f"{slug}.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    columns = _header(csv_path)
    factor_names = _factor_columns(metadata, factor_mode)
    requested = [*factor_names, "y"]
    missing = [name for name in requested if name not in columns]
    if missing:
        raise ValueError(f"{slug}: missing columns {missing}")

    usecols = [columns.index(name) for name in requested]
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=usecols)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    category_cols: list[np.ndarray] = []
    n_levels: list[int] = []
    for j in range(len(factor_names)):
        encoded, levels = _factorize(raw[:, j].astype(np.int64, copy=False))
        category_cols.append(encoded)
        n_levels.append(levels)

    categories = np.asfortranarray(np.column_stack(category_cols).astype(np.uint32, copy=False))
    y = np.ascontiguousarray(raw[:, -1], dtype=np.float64)
    return categories, y, n_levels


def max_abs_group_mean(
    categories: np.ndarray,
    n_levels: list[int],
    demeaned: np.ndarray,
) -> float:
    worst = 0.0
    for j, levels in enumerate(n_levels):
        sums = np.zeros(levels, dtype=np.float64)
        counts = np.zeros(levels, dtype=np.float64)
        cats = categories[:, j]
        np.add.at(sums, cats, demeaned)
        np.add.at(counts, cats, 1.0)
        means = sums / np.maximum(counts, 1.0)
        worst = max(worst, float(np.abs(means).max(initial=0.0)))
    return worst


def run_dataset(data_dir: Path, slug: str, factor_mode: str, tol: float, maxiter: int) -> None:
    categories, y, n_levels = load_problem(data_dir, slug, factor_mode)
    options = within.LsmrOptions(tol=tol, maxiter=maxiter)

    solver = within.Solver(categories)
    approx = solver.solve_approx_parallel(y, options)
    exact = solver.solve(y, options)

    denom = max(float(np.linalg.norm(exact.demeaned)), 1e-15)
    rel_demeaned_gap = float(np.linalg.norm(approx.demeaned - exact.demeaned) / denom)
    approx_group_mean = max_abs_group_mean(categories, n_levels, approx.demeaned)
    exact_group_mean = max_abs_group_mean(categories, n_levels, exact.demeaned)

    print(
        f"{slug:28s} rows={categories.shape[0]:8d} dofs={sum(n_levels):8d} "
        f"approx_resid={approx.residual:9.2e} exact_resid={exact.residual:9.2e} "
        f"approx_time={approx.time_solve:7.3f}s exact_time={exact.time_solve:7.3f}s "
        f"rel_gap={rel_demeaned_gap:9.2e} "
        f"group_mean={approx_group_mean:9.2e}/{exact_group_mean:9.2e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "all-csv",
        help="Directory created by unzipping benchmarks/all-csv.zip.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["credit2", "soccer", "synthetic-zigzag"],
        help="CSV benchmark slugs to run.",
    )
    parser.add_argument(
        "--factors",
        choices=["graph", "all"],
        default="graph",
        help="Use graph_id1/graph_id2 only, or include additional identifier columns.",
    )
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=2000)
    args = parser.parse_args()

    for slug in args.datasets:
        run_dataset(args.data_dir, slug, args.factors, args.tol, args.maxiter)


if __name__ == "__main__":
    main()
