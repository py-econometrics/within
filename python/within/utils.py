"""Utility functions for generating synthetic panel data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_akm_panel(
    n_workers: int = 20_000,
    n_firms: int = 2_000,
    n_years: int = 10,
    avg_spells: float = 2.5,
    seed: int = 42,
) -> dict[str, NDArray]:
    """Generate a matched employer-employee panel with job mobility.

    Workers move between firms in local labor-market clusters, producing
    a realistic three-way (firm, worker, year) fixed-effects structure.

    Parameters
    ----------
    n_workers : int
        Number of distinct workers.
    n_firms : int
        Number of distinct firms.
    n_years : int
        Number of time periods.
    avg_spells : float
        Average number of firm spells per worker.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        ``fe``  – uint32 array of shape (n, 3) with columns [firm, worker, year]
        ``y``   – float64 response vector (log wages)
        ``X``   – float64 covariate matrix (n, 2): experience and a noise regressor
        ``beta_true`` – true coefficient vector
    """
    rng = np.random.default_rng(seed)

    worker_ids, firm_ids, year_ids = [], [], []
    cluster_size = max(1, n_firms // 200)

    for w in range(n_workers):
        k = max(1, rng.poisson(avg_spells - 1) + 1)
        cluster = rng.integers(0, max(1, n_firms // cluster_size))
        pool = np.arange(
            cluster * cluster_size,
            min((cluster + 1) * cluster_size, n_firms),
        )
        spell_firms = rng.choice(pool, size=k, replace=True)
        spell_lengths = rng.multinomial(n_years, np.ones(k) / k)

        t = 0
        for firm, length in zip(spell_firms, spell_lengths):
            for yr in range(t, min(t + length, n_years)):
                worker_ids.append(w)
                firm_ids.append(firm)
                year_ids.append(yr)
            t += length

    firm_ids = np.asarray(firm_ids, dtype=np.uint32)
    worker_ids = np.asarray(worker_ids, dtype=np.uint32)
    year_ids = np.asarray(year_ids, dtype=np.uint32)
    fe = np.column_stack([firm_ids, worker_ids, year_ids])

    n = len(firm_ids)
    beta_true = np.array([0.05, 0.02])
    X = np.column_stack([
        rng.standard_normal(n),         # e.g. log(hours)
        rng.standard_normal(n) * 0.5,   # e.g. tenure proxy
    ])
    y = (
        X @ beta_true
        + rng.standard_normal(n_firms)[firm_ids] * 0.3
        + rng.standard_normal(n_workers)[worker_ids] * 0.5
        + 0.02 * np.arange(n_years, dtype=float)[year_ids]
        + 0.1 * rng.standard_normal(n)
    )

    return {"fe": fe, "y": y, "X": X, "beta_true": beta_true}