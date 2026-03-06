from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    OperatorRepr,
    SolveResult,
    LSMR,
    CG,
    GMRES,
    AdditiveSchwarz,
    MultiplicativeSchwarz,
    py_solve as _py_solve,
)


def solve(categories, y, config=None, *, n_levels=None, weights=None, layout=None):
    """Solve fixed-effects normal equations.

    categories can be a 2D uintp array (n_obs, n_factors) or a list of 1D
    int64 arrays (one per factor).
    """
    import numpy as np

    if config is None:
        config = CG(preconditioner=AdditiveSchwarz())

    if isinstance(categories, list):
        categories = np.column_stack(categories).astype(np.uintp)
    return _py_solve(
        categories, y, config, n_levels=n_levels, weights=weights, layout=layout
    )


__all__ = [
    "ApproxCholConfig",
    "ApproxSchurConfig",
    "OperatorRepr",
    "SolveResult",
    "LSMR",
    "CG",
    "GMRES",
    "AdditiveSchwarz",
    "MultiplicativeSchwarz",
    "solve",
]
