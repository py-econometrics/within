from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_synthetic_data(
    n_levels: list[int],
    n_rows: int,
    seed: int = 42,
) -> tuple[list[NDArray[np.int64]], NDArray[np.float64], NDArray[np.float64]]:
    """Generate synthetic fixed-effects data: y = D @ x_true (no noise)."""
    rng = np.random.default_rng(seed)
    cats = [rng.integers(0, nl, size=n_rows) for nl in n_levels]
    x_true = rng.standard_normal(sum(n_levels))
    y = np.zeros(n_rows)
    offset = 0
    for f, nl in enumerate(n_levels):
        y += x_true[offset + cats[f]]
        offset += nl
    return cats, x_true, y
