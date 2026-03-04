"""Shared panel-data simulation logic for AKM-style benchmarks.

Provides helpers for generating realistic matched employer-employee panel
data with the structural pathologies that make real AKM estimation hard:

1. Power-law (Zipf) firm-size distributions
2. Low worker mobility (sparse bipartite graph)
3. Near-disconnected regional clusters
4. Combinations of the above
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._problems import _make_response


def zipf_firm_sizes(
    n_firms: int, exponent: float,
) -> NDArray[np.float64]:
    """Return Zipf-distributed firm probability weights.

    ``weights[i] ~ (i+1)^{-exponent}``, normalized to sum to 1.
    """
    ranks = np.arange(1, n_firms + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    weights /= weights.sum()
    return weights


def simulate_mobility(
    initial_firm: NDArray[np.intp],
    n_years: int,
    mobility_rate: float,
    firm_weights: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    cluster_map: NDArray[np.intp] | None = None,
    cross_cluster_rate: float = 0.0,
) -> NDArray[np.intp]:
    """Simulate year-by-year job transitions for a panel.

    Parameters
    ----------
    initial_firm
        Shape ``(n_workers,)`` — each worker's firm in year 0.
    n_years
        Number of panel years.
    mobility_rate
        Probability a worker switches firm each year.
    firm_weights
        Probability weights for firm choice (length = n_firms).
    rng
        NumPy random generator.
    cluster_map
        If given, shape ``(n_firms,)`` mapping each firm to a cluster id.
        Moves stay within the worker's current cluster unless a cross-cluster
        move is triggered.
    cross_cluster_rate
        Fraction of moves that jump across clusters (only used when
        *cluster_map* is not None).

    Returns
    -------
    NDArray[np.intp]
        Shape ``(n_workers, n_years)`` — firm id for each worker-year.
    """
    n_workers = len(initial_firm)
    n_firms = len(firm_weights)
    assignments = np.empty((n_workers, n_years), dtype=np.intp)
    assignments[:, 0] = initial_firm

    # Pre-compute per-cluster firm weights for clustered mobility
    cluster_firms: list[NDArray[np.intp]] = []
    cluster_weights: list[NDArray[np.float64]] = []
    n_clusters = 0
    if cluster_map is not None:
        n_clusters = int(cluster_map.max()) + 1
        for c in range(n_clusters):
            mask = cluster_map == c
            firms_c = np.where(mask)[0].astype(np.intp)
            w_c = firm_weights[mask].copy()
            w_c /= w_c.sum() if w_c.sum() > 0 else 1.0
            cluster_firms.append(firms_c)
            cluster_weights.append(w_c)

    for t in range(1, n_years):
        prev = assignments[:, t - 1]
        # Who moves this year?
        moves = rng.random(n_workers) < mobility_rate
        n_movers = moves.sum()
        new_firms = prev.copy()

        if n_movers == 0:
            assignments[:, t] = new_firms
            continue

        if cluster_map is not None:
            # Clustered mobility: most moves within cluster
            mover_idx = np.where(moves)[0]
            mover_clusters = cluster_map[prev[mover_idx]]
            cross = rng.random(n_movers) < cross_cluster_rate

            # Cross-cluster moves: sample from all firms
            n_cross = cross.sum()
            if n_cross > 0:
                cross_idx = mover_idx[cross]
                new_firms[cross_idx] = rng.choice(
                    n_firms, size=n_cross, p=firm_weights,
                ).astype(np.intp)

            # Within-cluster moves
            within_idx = mover_idx[~cross]
            within_clusters = mover_clusters[~cross]
            for c in range(n_clusters):
                mask_c = within_clusters == c
                n_c = mask_c.sum()
                if n_c == 0:
                    continue
                idx_c = within_idx[mask_c]
                new_firms[idx_c] = rng.choice(
                    cluster_firms[c], size=n_c, p=cluster_weights[c],
                )
        else:
            # Unclustered: sample from all firms weighted by size
            mover_idx = np.where(moves)[0]
            new_firms[mover_idx] = rng.choice(
                n_firms, size=n_movers, p=firm_weights,
            ).astype(np.intp)

        assignments[:, t] = new_firms

    return assignments


def find_largest_component(
    worker_ids: NDArray[np.intp], firm_ids: NDArray[np.intp],
) -> NDArray[np.bool_]:
    """Return boolean mask for observations in the largest connected component.

    Uses union-find on the bipartite worker-firm graph (no scipy dependency).
    """
    n_workers = int(worker_ids.max()) + 1
    n_firms = int(firm_ids.max()) + 1
    n = n_workers + n_firms

    # Union-find
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Build bipartite edges: workers [0, n_workers), firms [n_workers, n_workers+n_firms)
    for w, f in zip(worker_ids, firm_ids):
        union(int(w), n_workers + int(f))

    # Find component for each worker
    worker_labels = np.array([find(i) for i in range(n_workers)], dtype=np.intp)

    # Label per observation
    obs_labels = worker_labels[worker_ids]

    # Find the largest component by number of observations
    unique, counts = np.unique(obs_labels, return_counts=True)
    largest = unique[counts.argmax()]

    return obs_labels == largest


def panel_to_design(
    firm_assignments: NDArray[np.intp],
    n_fe: int,
    rng: np.random.Generator,
    *,
    prune_components: bool = False,
) -> tuple[list[NDArray[np.int64]], list[int], NDArray[np.float64]]:
    """Convert a panel of firm assignments to categories and response.

    Parameters
    ----------
    firm_assignments
        Shape ``(n_workers, n_years)`` — firm id per worker-year.
    n_fe
        Number of fixed effects (2 = worker+firm, 3 = worker+firm+year).
    rng
        Random generator for response variable.
    prune_components
        If True, restrict to the largest connected component and
        re-index worker/firm ids to contiguous ranges.

    Returns
    -------
    (categories, n_levels, y)
        The category arrays, level counts, and response vector.
    """
    n_workers, n_years = firm_assignments.shape

    # Flatten panel to observation-level arrays
    worker_ids = np.repeat(np.arange(n_workers, dtype=np.intp), n_years)
    year_ids = np.tile(np.arange(n_years, dtype=np.intp), n_workers)
    firm_ids = firm_assignments.ravel().astype(np.intp)

    if prune_components:
        keep = find_largest_component(worker_ids, firm_ids)
        worker_ids = worker_ids[keep]
        year_ids = year_ids[keep]
        firm_ids = firm_ids[keep]

        # Re-index to contiguous [0, n)
        worker_ids = _reindex(worker_ids)
        firm_ids = _reindex(firm_ids)
        year_ids = _reindex(year_ids)

    n_workers_final = int(worker_ids.max()) + 1
    n_firms_final = int(firm_ids.max()) + 1
    n_years_final = int(year_ids.max()) + 1

    if n_fe == 2:
        categories = [worker_ids.astype(np.int64), firm_ids.astype(np.int64)]
        n_levels = [n_workers_final, n_firms_final]
    else:
        categories = [worker_ids.astype(np.int64), firm_ids.astype(np.int64), year_ids.astype(np.int64)]
        n_levels = [n_workers_final, n_firms_final, n_years_final]

    y = _make_response(categories, n_levels, rng)
    return categories, n_levels, y


def _reindex(ids: NDArray[np.intp]) -> NDArray[np.intp]:
    """Map arbitrary integer ids to contiguous 0-based indices."""
    _, inverse = np.unique(ids, return_inverse=True)
    return inverse.astype(np.intp)
