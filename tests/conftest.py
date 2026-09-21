from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from within import PreconditionerConfig

_MAPS = [
    PreconditionerConfig.Diagonal(),
    PreconditionerConfig.Additive(),
    PreconditionerConfig.Adaptive(),
]

# Variants that hold a map a Solver can hand back, and every variant including Off.
every_preconditioner_map = pytest.mark.parametrize(
    "precond", _MAPS, ids=lambda p: type(p).__name__
)
every_preconditioner = pytest.mark.parametrize(
    "precond", [PreconditionerConfig.Off(), *_MAPS], ids=lambda p: type(p).__name__
)


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


def as_solver_categories(cats):
    """Stack per-factor level arrays into F-contiguous uint32 (the solver's fast path)."""
    return np.asfortranarray(np.column_stack(cats).astype(np.uint32))
