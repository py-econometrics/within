"""Benchmark one-shot Schwarz residualization on Correia CSV datasets.

Usage:

    uv run python benchmarks/approx_parallel_csv.py --output results.csv

The script expects ``benchmarks/all-csv`` to contain the CSV files and metadata
JSONs from ``all-csv.zip``. It compares one-shot approximate residualization to
the corrected LSMR solve using the same cached preconditioner. It also computes
cheap graph diagnostics on the two-way benchmark graph:

- connected components;
- largest-component edge share;
- approximate second eigenvalue of the normalized Laplacian;
- conductance of the best Fiedler-vector sweep cut.

The final ``recommended_solver`` column is a routing recommendation, not a
proof of estimator validity. It combines graph diagnostics with the observed
one-shot residual and coefficient sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

import within


@dataclass
class GraphDiagnostics:
    n_nodes: int
    n_unique_edges: int
    n_components: int
    largest_component_nodes: int
    largest_component_node_share: float
    largest_component_edge_share: float
    lambda2_lcc: float
    fiedler_phi: float
    fiedler_balance: float
    spectral_status: str


@dataclass
class BenchmarkRow:
    dataset: str
    rows: int
    dofs: int
    components: int
    lcc_node_share: float
    lcc_edge_share: float
    lambda2_lcc: float
    phi_fiedler: float
    fiedler_balance: float
    approx_residual_max: float
    corrected_residual_max: float
    rel_gap_y: float
    beta_rel_gap: float
    setup_time: float
    approx_time: float
    corrected_time: float
    recommended_solver: str
    spectral_status: str


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readline().strip().split(",")


def _factorize(values: np.ndarray) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.uint32, copy=False), int(inverse.max(initial=-1) + 1)


def _factor_columns(metadata: dict[str, Any], mode: str) -> list[str]:
    graph_cols = [metadata["graph_id1"], metadata["graph_id2"]]
    if mode == "graph":
        return graph_cols
    extras = metadata.get("additional_identifier_columns", [])
    return [*graph_cols, *extras]


def discover_datasets(data_dir: Path) -> list[str]:
    metadata_dir = data_dir / "metadata"
    return sorted(path.stem for path in metadata_dir.glob("*.json"))


def load_problem(
    data_dir: Path,
    slug: str,
    factor_mode: str,
    rhs_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    metadata_path = data_dir / "metadata" / f"{slug}.json"
    csv_path = data_dir / f"{slug}.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    columns = _header(csv_path)
    factor_names = _factor_columns(metadata, factor_mode)
    rhs_names = ["y"] if rhs_mode == "y" else ["y", "x1", "x2"]
    requested = [*factor_names, *rhs_names]
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

    categories = np.asfortranarray(
        np.column_stack(category_cols).astype(np.uint32, copy=False)
    )
    rhs = np.asfortranarray(raw[:, -len(rhs_names) :].astype(np.float64, copy=False))
    return categories, rhs, n_levels, metadata


def _build_adjacency(categories: np.ndarray, n_levels: list[int]) -> sparse.csr_matrix:
    if categories.shape[1] < 2:
        raise ValueError("spectral graph diagnostics require at least two factors")

    n_left, n_right = n_levels[0], n_levels[1]
    weights = np.ones(categories.shape[0], dtype=np.float64)
    cross = sparse.coo_matrix(
        (weights, (categories[:, 0], categories[:, 1])),
        shape=(n_left, n_right),
    ).tocsr()
    cross.sum_duplicates()

    zero_left = sparse.csr_matrix((n_left, n_left), dtype=np.float64)
    zero_right = sparse.csr_matrix((n_right, n_right), dtype=np.float64)
    return sparse.vstack(
        [
            sparse.hstack([zero_left, cross], format="csr"),
            sparse.hstack([cross.T, zero_right], format="csr"),
        ],
        format="csr",
    )


def _component_edge_share(adjacency: sparse.csr_matrix, labels: np.ndarray, label: int) -> float:
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    total = float(degrees.sum() / 2.0)
    if total <= 0.0:
        return 0.0
    component_weight = float(degrees[labels == label].sum() / 2.0)
    return component_weight / total


def _normalized_adjacency_operator(
    adjacency: sparse.csr_matrix,
    degrees: np.ndarray,
) -> LinearOperator:
    inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    positive = degrees > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])

    def matvec(v: np.ndarray) -> np.ndarray:
        return inv_sqrt * (adjacency @ (inv_sqrt * v))

    return LinearOperator(adjacency.shape, matvec=matvec, dtype=np.float64)


def _dense_normalized_adjacency(
    adjacency: sparse.csr_matrix,
    degrees: np.ndarray,
) -> np.ndarray:
    dense = adjacency.toarray()
    inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    positive = degrees > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    return inv_sqrt[:, None] * dense * inv_sqrt[None, :]


def _fiedler_pair(
    adjacency: sparse.csr_matrix,
    eigen_tol: float,
    eigen_maxiter: int,
) -> tuple[float, np.ndarray, str]:
    n_nodes = adjacency.shape[0]
    if n_nodes <= 1:
        return math.nan, np.zeros(n_nodes, dtype=np.float64), "singleton"

    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    if n_nodes <= 3:
        sym_adj = _dense_normalized_adjacency(adjacency, degrees)
        values, vectors = np.linalg.eigh(sym_adj)
        order = np.argsort(values)[::-1]
        if len(order) < 2:
            return math.nan, np.zeros(n_nodes, dtype=np.float64), "too_small"
        mu2 = float(values[order[1]])
        return 1.0 - mu2, vectors[:, order[1]], "dense"

    op = _normalized_adjacency_operator(adjacency, degrees)
    try:
        values, vectors = eigsh(
            op,
            k=2,
            which="LA",
            tol=eigen_tol,
            maxiter=eigen_maxiter,
            ncv=20,
        )
        status = "ok"
    except ArpackNoConvergence as err:
        values = err.eigenvalues
        vectors = err.eigenvectors
        if values is None or vectors is None or len(values) < 2:
            return math.nan, np.zeros(n_nodes, dtype=np.float64), "no_convergence"
        status = "partial"

    order = np.argsort(values)[::-1]
    mu2 = float(values[order[1]])
    lambda2 = min(max(1.0 - mu2, 0.0), 2.0)
    return lambda2, vectors[:, order[1]], status


def _sweep_conductance(
    adjacency: sparse.csr_matrix,
    fiedler: np.ndarray,
) -> tuple[float, float]:
    n_nodes = adjacency.shape[0]
    if n_nodes <= 1 or fiedler.size != n_nodes:
        return math.nan, math.nan

    order = np.argsort(fiedler, kind="mergesort")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    total_volume = float(degrees.sum())
    if total_volume <= 0.0:
        return math.nan, math.nan

    in_set = np.zeros(n_nodes, dtype=bool)
    cut = 0.0
    volume = 0.0
    best_phi = math.inf
    best_balance = math.nan
    indptr = adjacency.indptr
    indices = adjacency.indices
    data = adjacency.data

    for step, node in enumerate(order[:-1], start=1):
        start, end = indptr[node], indptr[node + 1]
        neighbors = indices[start:end]
        weights = data[start:end]
        inside_weight = float(weights[in_set[neighbors]].sum())
        degree = float(degrees[node])

        cut += degree - 2.0 * inside_weight
        volume += degree
        in_set[node] = True

        denom = min(volume, total_volume - volume)
        if denom <= 0.0:
            continue
        phi = max(cut, 0.0) / denom
        if phi < best_phi:
            best_phi = phi
            best_balance = denom / total_volume

    if not math.isfinite(best_phi):
        return math.nan, math.nan
    return best_phi, best_balance


def spectral_diagnostics(
    categories: np.ndarray,
    n_levels: list[int],
    eigen_tol: float,
    eigen_maxiter: int,
) -> GraphDiagnostics:
    adjacency = _build_adjacency(categories, n_levels)
    n_components, labels = csgraph.connected_components(
        adjacency, directed=False, return_labels=True
    )
    counts = np.bincount(labels)
    lcc_label = int(np.argmax(counts))
    lcc_nodes = int(counts[lcc_label])
    lcc_node_share = lcc_nodes / float(adjacency.shape[0])
    lcc_edge_share = _component_edge_share(adjacency, labels, lcc_label)

    lcc_index = np.flatnonzero(labels == lcc_label)
    lcc_adjacency = adjacency[lcc_index][:, lcc_index].tocsr()
    lambda2, fiedler, status = _fiedler_pair(
        lcc_adjacency,
        eigen_tol=eigen_tol,
        eigen_maxiter=eigen_maxiter,
    )
    phi, balance = _sweep_conductance(lcc_adjacency, fiedler)

    return GraphDiagnostics(
        n_nodes=int(adjacency.shape[0]),
        n_unique_edges=int(adjacency.nnz // 2),
        n_components=int(n_components),
        largest_component_nodes=lcc_nodes,
        largest_component_node_share=lcc_node_share,
        largest_component_edge_share=lcc_edge_share,
        lambda2_lcc=lambda2,
        fiedler_phi=phi,
        fiedler_balance=balance,
        spectral_status=status,
    )


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


def _beta_from_residuals(demeaned: np.ndarray) -> np.ndarray:
    y = demeaned[:, 0]
    x = demeaned[:, 1:]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _max(values: list[float] | tuple[float, ...]) -> float:
    return float(max(values)) if values else math.nan


def _solver_recommendation(
    diagnostics: GraphDiagnostics,
    approx_residual: float,
    beta_rel_gap: float,
) -> str:
    beta_ok = not math.isfinite(beta_rel_gap) or beta_rel_gap <= 1e-2
    approx_ok = approx_residual <= 1e-3 and beta_ok

    if diagnostics.n_components > 1 and diagnostics.largest_component_edge_share < 0.95:
        return "component-parallel exact"

    if approx_ok:
        return "one-shot parallel"

    low_gap = math.isfinite(diagnostics.lambda2_lcc) and diagnostics.lambda2_lcc <= 1e-3
    low_phi = math.isfinite(diagnostics.fiedler_phi) and diagnostics.fiedler_phi <= 1e-2
    if low_gap or low_phi or diagnostics.n_components > 1:
        return "Schwarz-preconditioned exact"

    return "MAP or accelerated MAP"


def run_dataset(
    data_dir: Path,
    slug: str,
    factor_mode: str,
    rhs_mode: str,
    tol: float,
    maxiter: int,
    eigen_tol: float,
    eigen_maxiter: int,
    skip_solve_components_above: int,
) -> BenchmarkRow:
    print(f"[{slug}] loading", file=sys.stderr, flush=True)
    categories, rhs, n_levels, _metadata = load_problem(data_dir, slug, factor_mode, rhs_mode)
    print(f"[{slug}] spectral diagnostics", file=sys.stderr, flush=True)
    diagnostics = spectral_diagnostics(
        categories,
        n_levels,
        eigen_tol=eigen_tol,
        eigen_maxiter=eigen_maxiter,
    )

    if diagnostics.n_components > skip_solve_components_above:
        print(
            f"[{slug}] skipping one-shot timing because components="
            f"{diagnostics.n_components} exceeds {skip_solve_components_above}",
            file=sys.stderr,
            flush=True,
        )
        return BenchmarkRow(
            dataset=slug,
            rows=int(categories.shape[0]),
            dofs=int(sum(n_levels)),
            components=diagnostics.n_components,
            lcc_node_share=diagnostics.largest_component_node_share,
            lcc_edge_share=diagnostics.largest_component_edge_share,
            lambda2_lcc=diagnostics.lambda2_lcc,
            phi_fiedler=diagnostics.fiedler_phi,
            fiedler_balance=diagnostics.fiedler_balance,
            approx_residual_max=math.nan,
            corrected_residual_max=math.nan,
            rel_gap_y=math.nan,
            beta_rel_gap=math.nan,
            setup_time=math.nan,
            approx_time=math.nan,
            corrected_time=math.nan,
            recommended_solver="component-parallel exact",
            spectral_status=f"{diagnostics.spectral_status}; solve_skipped_many_components",
        )

    options = within.LsmrOptions(tol=tol, maxiter=maxiter)

    print(f"[{slug}] building solver", file=sys.stderr, flush=True)
    setup_start = time.perf_counter()
    solver = within.Solver(categories)
    setup_time = time.perf_counter() - setup_start

    print(f"[{slug}] one-shot solve", file=sys.stderr, flush=True)
    approx_start = time.perf_counter()
    approx = solver.solve_approx_parallel_batch(rhs, options)
    approx_time = time.perf_counter() - approx_start

    print(f"[{slug}] corrected solve", file=sys.stderr, flush=True)
    corrected_start = time.perf_counter()
    corrected = solver.solve_batch(rhs, options)
    corrected_time = time.perf_counter() - corrected_start

    denom_y = max(float(np.linalg.norm(corrected.demeaned[:, 0])), 1e-15)
    rel_gap_y = float(np.linalg.norm(approx.demeaned[:, 0] - corrected.demeaned[:, 0]) / denom_y)

    if corrected.demeaned.shape[1] >= 3:
        beta_exact = _beta_from_residuals(corrected.demeaned)
        beta_approx = _beta_from_residuals(approx.demeaned)
        beta_rel_gap = float(
            np.linalg.norm(beta_approx - beta_exact)
            / max(float(np.linalg.norm(beta_exact)), 1e-15)
        )
    else:
        beta_rel_gap = math.nan

    approx_residual = _max(approx.residual)
    corrected_residual = _max(corrected.residual)
    recommended = _solver_recommendation(diagnostics, approx_residual, beta_rel_gap)

    return BenchmarkRow(
        dataset=slug,
        rows=int(categories.shape[0]),
        dofs=int(sum(n_levels)),
        components=diagnostics.n_components,
        lcc_node_share=diagnostics.largest_component_node_share,
        lcc_edge_share=diagnostics.largest_component_edge_share,
        lambda2_lcc=diagnostics.lambda2_lcc,
        phi_fiedler=diagnostics.fiedler_phi,
        fiedler_balance=diagnostics.fiedler_balance,
        approx_residual_max=approx_residual,
        corrected_residual_max=corrected_residual,
        rel_gap_y=rel_gap_y,
        beta_rel_gap=beta_rel_gap,
        setup_time=setup_time,
        approx_time=approx_time,
        corrected_time=corrected_time,
        recommended_solver=recommended,
        spectral_status=diagnostics.spectral_status,
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _format_seconds(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3f}s"


def print_row(row: BenchmarkRow) -> None:
    print(
        f"{row.dataset:28s} rows={row.rows:8d} dofs={row.dofs:8d} "
        f"comp={row.components:6d} lcc={row.lcc_edge_share:6.3f} "
        f"lambda2={_format_float(row.lambda2_lcc):>9s} "
        f"phi={_format_float(row.phi_fiedler):>9s} "
        f"approx_resid={row.approx_residual_max:9.2e} "
        f"exact_resid={row.corrected_residual_max:9.2e} "
        f"rel_gap={row.rel_gap_y:9.2e} "
        f"approx_time={row.approx_time:7.3f}s "
        f"exact_time={row.corrected_time:7.3f}s "
        f"solver={row.recommended_solver}",
        flush=True,
    )


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Dataset",
        "Rows",
        "DOFs",
        "Comp.",
        "LCC edge share",
        "$\\lambda_2$",
        "$\\phi(S)$",
        "Approx resid.",
        "Rel. y gap",
        "One-shot time",
        "Corrected time",
        "Solver",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|"
        + "|".join(
            [
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---",
            ]
        )
        + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.dataset}`",
                    f"{row.rows:,}",
                    f"{row.dofs:,}",
                    f"{row.components:,}",
                    _format_float(row.lcc_edge_share),
                    _format_float(row.lambda2_lcc),
                    _format_float(row.phi_fiedler),
                    _format_float(row.approx_residual_max),
                    _format_float(row.rel_gap_y),
                    _format_seconds(row.approx_time),
                    _format_seconds(row.corrected_time),
                    row.recommended_solver,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default=None,
        help="CSV benchmark slugs to run. Defaults to every metadata entry.",
    )
    parser.add_argument(
        "--factors",
        choices=["graph", "all"],
        default="graph",
        help="Use graph_id1/graph_id2 only, or include additional identifier columns.",
    )
    parser.add_argument(
        "--rhs",
        choices=["y", "yx"],
        default="y",
        help="Residualize only y, or y plus x1/x2 for coefficient sensitivity.",
    )
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--eigen-tol", type=float, default=1e-4)
    parser.add_argument("--eigen-maxiter", type=int, default=1000)
    parser.add_argument(
        "--skip-solve-components-above",
        type=int,
        default=50_000,
        help=(
            "Skip current Schwarz timing when a graph has more connected "
            "components than this. The diagnostic still runs and the solver "
            "recommendation is component-parallel exact."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional Markdown table output path.",
    )
    args = parser.parse_args()

    datasets = args.datasets or discover_datasets(args.data_dir)
    rows: list[BenchmarkRow] = []
    for slug in datasets:
        row = run_dataset(
            args.data_dir,
            slug,
            args.factors,
            args.rhs,
            args.tol,
            args.maxiter,
            args.eigen_tol,
            args.eigen_maxiter,
            args.skip_solve_components_above,
        )
        rows.append(row)
        print_row(row)

    if args.output:
        write_csv(args.output, rows)
    if args.markdown_output:
        write_markdown(args.markdown_output, rows)


if __name__ == "__main__":
    main()
