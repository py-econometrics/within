"""Console table formatting for benchmark results."""

from __future__ import annotations

from ._types import BenchmarkResult


def print_table(
    results: list[BenchmarkResult],
    columns: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Print benchmark results as a formatted console table."""
    if not results:
        print("  (no results)")
        return

    if columns is None:
        columns = [
            "config",
            "setup_time", "solve_time", "iterations",
            "final_residual", "converged",
        ]

    _COL_FMT: dict[str, tuple[str, int, object]] = {
        "problem":        ("Problem",    28, lambda r: r.problem),
        "config":         ("Config",     16, lambda r: r.config),
        "n_dofs":         ("DOFs",        8, lambda r: r.n_dofs),
        "n_rows":         ("Rows",        8, lambda r: r.n_rows),
        "setup_time":     ("Setup(s)",    9, lambda r: f"{r.setup_time:.4f}"),
        "solve_time":     ("Solve(s)",    9, lambda r: f"{r.solve_time:.4f}"),
        "total_time":     ("Total(s)",    9, lambda r: f"{r.setup_time + r.solve_time:.4f}"),
        "iterations":     ("Iters",       6, lambda r: r.iterations),
        "final_residual": ("Residual",   12, lambda r: f"{r.final_residual:.2e}"),
        "converged":      ("Conv",        5, lambda r: "OK" if r.converged else "FAIL"),
        "passed":         ("Check",       6, lambda r: "PASS" if r.passed else "FAIL" if r.passed is not None else "--"),
    }

    if title:
        print(f"\n{title}")
        print("=" * 100)

    # Build header
    header_parts = []
    for col in columns:
        label, width, _ = _COL_FMT[col]
        header_parts.append(f"{label:>{width}}")
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))

    # Build rows
    for r in results:
        parts = []
        for col in columns:
            _, width, extractor = _COL_FMT[col]
            val = extractor(r)
            if col in ("problem", "config"):
                parts.append(f"{val:<{width}}")
            else:
                parts.append(f"{str(val):>{width}}")
        print(" ".join(parts))


def print_pivot(
    results: list[BenchmarkResult],
    row: str = "problem",
    col: str = "config",
    value: str = "iterations",
) -> None:
    """Print a pivot table of results.

    *row* and *col* select which ``BenchmarkResult`` fields define the axes.
    *value* selects the field to display in cells.
    """
    if not results:
        print("  (no results)")
        return

    # Collect unique row/col values preserving insertion order
    row_vals: list[str] = []
    col_vals: list[str] = []
    seen_rows: set[str] = set()
    seen_cols: set[str] = set()
    lookup: dict[tuple[str, str], BenchmarkResult] = {}

    for r in results:
        rv = str(getattr(r, row))
        cv = str(getattr(r, col))
        if rv not in seen_rows:
            row_vals.append(rv)
            seen_rows.add(rv)
        if cv not in seen_cols:
            col_vals.append(cv)
            seen_cols.add(cv)
        lookup[(rv, cv)] = r

    col_w = max(max(len(c) for c in col_vals) + 1, 10)
    row_w = max(max(len(r) for r in row_vals) + 1, 28)

    # Header
    header = f"{'':>{row_w}}"
    for c in col_vals:
        header += f" {c:>{col_w}}"
    print(header)
    print("-" * len(header))

    # Rows
    for rv in row_vals:
        line = f"{rv:<{row_w}}"
        for cv in col_vals:
            r = lookup.get((rv, cv))
            if r is None:
                line += f" {'--':>{col_w}}"
            else:
                v = getattr(r, value)
                if isinstance(v, float):
                    cell = f"{v:.4f}" if v > 0.01 else f"{v:.2e}"
                elif isinstance(v, bool):
                    cell = "Y" if v else "N"
                else:
                    cell = str(v)
                if not r.converged and value == "iterations":
                    cell += "*"
                line += f" {cell:>{col_w}}"
        print(line)
