from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from within import Effect, LsmrOptions, PreconditionerConfig, Solver

from .suites.akm_slopes import _LADDER, _SPECS, _mover_panel, _slope_case

Array = NDArray[np.float64]


def inverse(matrix: Array) -> Array:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    keep = values > max(values[-1] * 1e-12, 1e-14)
    return (vectors[:, keep] / values[keep]) @ vectors[:, keep].T


def spectrum(gram: Array, preconditioner: Array) -> dict[str, object]:
    values, vectors = np.linalg.eigh((preconditioner + preconditioner.T) * 0.5)
    keep = values > max(values[-1] * 1e-13, 1e-15)
    factor = vectors[:, keep] * np.sqrt(values[keep])
    eigenvalues = np.linalg.eigvalsh(factor.T @ gram @ factor)
    cutoff = max(eigenvalues[-1] * 1e-11, 1e-13)
    positive = eigenvalues[eigenvalues > cutoff]
    return {
        "rank": len(positive),
        "cutoff": cutoff,
        "condition": float(positive[-1] / positive[0]),
        "smallest": positive[:6].tolist(),
        "largest": float(positive[-1]),
    }


def dense_term(ids: NDArray[np.uint32], slope: Array) -> Array:
    n = int(ids.max()) + 1
    count = np.bincount(ids, minlength=n)
    mean = np.bincount(ids, weights=slope, minlength=n) / count
    centered = slope - mean[ids]
    norms = np.sqrt(np.bincount(ids, weights=centered * centered, minlength=n))
    loadings = centered / np.where(norms[ids] > 0, norms[ids], 1)
    design = np.zeros((len(ids), 2 * n))
    design[np.arange(len(ids)), ids] = 1
    design[np.arange(len(ids)), ids + n] = loadings
    return design


def slow_modes(design: Array, maps: dict[str, Array]) -> dict[str, object]:
    _, singular, right = np.linalg.svd(design, full_matrices=False)
    keep = singular > singular[0] * 1e-12
    factor = right[keep].T * singular[keep]
    result: dict[str, object] = {}
    for name, matrix in maps.items():
        values = np.linalg.eigvalsh(factor.T @ matrix @ factor)
        result[name] = {
            "rank": len(values),
            "condition": float(values[-1] / values[0]),
            "below_max_over_10": int((values < values[-1] / 10).sum()),
            "below_max_over_100": int((values < values[-1] / 100).sum()),
        }
    return result


def audit(
    codes: list[NDArray[np.uint32]], slopes: list[Array], y: Array
) -> dict[str, object]:
    terms = [dense_term(ids, slope) for ids, slope in zip(codes, slopes)]
    design = np.column_stack(terms)
    gram = design.T @ design
    effects = [Effect(ids, True, [slope]) for ids, slope in zip(codes, slopes)]
    solver = Solver(effects, preconditioner=PreconditionerConfig.Additive())
    result = solver.solve(y, options=LsmrOptions(tol=1e-12, maxiter=20_000))
    preconditioner = np.column_stack(
        [solver.preconditioner.apply(e) for e in np.eye(design.shape[1])]
    )
    worker, firm = terms
    nw, nf = worker.shape[1] // 2, firm.shape[1] // 2
    exact = np.zeros_like(gram)
    for a in range(2):
        for b in range(2):
            indices = np.r_[np.arange(nw) + a * nw, np.arange(nf) + 2 * nw + b * nf]
            exact[np.ix_(indices, indices)] += inverse(gram[np.ix_(indices, indices)])
    projected = firm - worker @ inverse(worker.T @ worker) @ worker.T @ firm
    schur = projected.T @ projected
    diagonal = np.diag(1 / np.maximum(np.diag(schur), 1e-12))
    blocks = np.zeros_like(schur)
    for level in range(nf):
        indices = np.array([level, level + nf])
        blocks[np.ix_(indices, indices)] = inverse(schur[np.ix_(indices, indices)])
    lift = np.vstack((-inverse(worker.T @ worker) @ worker.T @ firm, np.eye(2 * nf)))
    correction = lift @ blocks @ lift.T
    auxiliary = Solver(
        [Effect(ids, False, [z]) for ids, z in zip(codes, slopes)],
        preconditioner=PreconditionerConfig.Additive(),
    )
    auxiliary_inverse = np.column_stack(
        [auxiliary.preconditioner.apply(e) for e in np.eye(nw + nf)]
    )
    transfer = np.zeros((design.shape[1], nw + nf))
    row_offset = column_offset = 0
    for ids, z, term in zip(codes, slopes, terms):
        n = term.shape[1] // 2
        norms = np.sqrt(np.bincount(ids, weights=z * z, minlength=n))
        coefficients = (term.T @ z).reshape(2, n)
        coefficients[0] /= np.bincount(ids, minlength=n)
        coefficients /= np.where(norms > 0, norms, 1)
        for j in range(n):
            transfer[row_offset + j, column_offset + j] = coefficients[0, j]
            transfer[row_offset + n + j, column_offset + j] = coefficients[1, j]
        row_offset += 2 * n
        column_offset += n
    patches = np.zeros_like(gram)
    edges = np.unique(np.column_stack(codes), axis=0)
    for w, f in edges:
        indices = np.array([w, w + nw, 2 * nw + f, 2 * nw + nf + f])
        patches[np.ix_(indices, indices)] += inverse(gram[np.ix_(indices, indices)])
    return {
        "dofs": design.shape[1],
        "native_iterations": result.iterations,
        "native_converged": result.converged,
        "channel_pairs": spectrum(gram, preconditioner),
        "exact_channel_pairs": spectrum(gram, exact),
        "schur_diagonal": spectrum(schur, diagonal),
        "schur_blocks": spectrum(schur, blocks),
        "pairs_plus_lifted_blocks": spectrum(gram, preconditioner + correction),
        "pairs_plus_raw_slope_pair": spectrum(
            gram, preconditioner + transfer @ auxiliary_inverse @ transfer.T
        ),
        "pairs_plus_edge_patches": spectrum(gram, preconditioner + patches),
        "slow_modes": slow_modes(
            design,
            {
                "pairs": preconditioner,
                "pairs_plus_blocks": preconditioner + correction,
            },
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--mobility", nargs="+", default=["1.0", "0.05"])
    parser.add_argument("--specs", nargs="+", default=[name for name, _ in _SPECS[1:]])
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
    elif args.rows > 5000:
        parser.error(
            "dense spectral audit is limited to 5000 rows; use --export for larger runs"
        )
    for seed in args.seeds:
        for mobility in args.mobility:
            panel = _mover_panel(args.rows, _LADDER[f"mobility={mobility}"], seed)
            for spec in args.specs:
                columns = dict(_SPECS)[spec]
                left, right = columns(panel, np.random.default_rng(seed + 1))
                if left is None or right is None:
                    parser.error("this probe requires a slope on each of two effects")
                slopes = [left[0], right[0]]
                if spec == "near_shared_year":
                    slopes[1] = slopes[0] + args.epsilon * np.random.default_rng(
                        seed + 2
                    ).standard_normal(len(slopes[0]))
                y = _slope_case(panel, columns, seed).y
                metadata = {
                    "rows": len(y),
                    "seed": seed,
                    "mobility": mobility,
                    "spec": spec,
                    "epsilon": args.epsilon,
                }
                if args.export:
                    name = (
                        f"n{args.rows}-s{seed}-m{mobility}-{spec}-e{args.epsilon}.csv"
                    )
                    path = args.export / name
                    np.savetxt(
                        path,
                        np.column_stack([*panel.codes[:2], *slopes, y]),
                        delimiter=",",
                        fmt=["%d", "%d", "%.17g", "%.17g", "%.17g"],
                        header="worker,firm,worker_slope,firm_slope,y",
                        comments="",
                    )
                    metadata["path"] = str(path)
                else:
                    metadata.update(audit(panel.codes[:2], slopes, y))
                print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
