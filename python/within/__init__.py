from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    SolveResult,
    LSMR,
    CG,
    GMRES,
    OneLevelSchwarz,
    MultiplicativeOneLevelSchwarz,
    py_solve as _py_solve,
    py_generate_synthetic_data as _py_generate_synthetic_data,
)


def solve(categories, y, config=None, *, n_levels=None, weights=None, layout=None):
    """Solve fixed-effects normal equations.

    categories can be a 2D uintp array (n_obs, n_factors) or a list of 1D
    int64 arrays (one per factor).
    """
    import numpy as np

    if config is None:
        config = CG()

    if isinstance(categories, list):
        categories = np.column_stack(categories).astype(np.uintp)
    return _py_solve(categories, y, config, n_levels=n_levels, weights=weights, layout=layout)


def generate_synthetic_data(n_levels_per_factor, n_rows, *, seed=42):
    """Generate synthetic fixed-effects data."""
    return _py_generate_synthetic_data(n_levels_per_factor, n_rows, seed=seed)


__all__ = [
    "ApproxCholConfig",
    "ApproxSchurConfig",
    "SolveResult",
    "LSMR",
    "CG",
    "GMRES",
    "OneLevelSchwarz",
    "MultiplicativeOneLevelSchwarz",
    "solve",
    "generate_synthetic_data",
]