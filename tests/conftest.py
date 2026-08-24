from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def populated_level_column(
    rng: np.random.Generator,
    n_levels: int,
    n_rows: int,
) -> NDArray[np.int64]:
    """Random level column in which every label ``0..n_levels`` occurs."""
    if not 0 < n_levels <= n_rows:
        raise ValueError("n_levels must be positive and no greater than n_rows")

    levels = np.concatenate(
        [
            np.arange(n_levels, dtype=np.int64),
            rng.integers(0, n_levels, size=n_rows - n_levels, dtype=np.int64),
        ]
    )
    rng.shuffle(levels)
    return levels


def generate_synthetic_data(
    n_levels: list[int],
    n_rows: int,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], NDArray[np.float64], NDArray[np.float64]]:
    """Generate synthetic fixed-effects data: y = D @ x_true (no noise)."""
    rng = np.random.default_rng(seed)
    cats = [populated_level_column(rng, nl, n_rows) for nl in n_levels]
    x_true = rng.standard_normal(sum(n_levels))
    y = np.zeros(n_rows)
    offset = 0
    for f, nl in enumerate(n_levels):
        y += x_true[offset + cats[f]]
        offset += nl
    return cats, x_true, y


def as_solver_categories(cats):
    """Stack per-factor level arrays into F-contiguous uint32 (the solver's fast path)."""
    return np.asfortranarray(np.column_stack(cats).astype(np.uint32))
