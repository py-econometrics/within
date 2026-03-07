from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray

from within._within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    _PyDemeanResult,
    _PySolveResult,
    LSMR,
    CG,
    GMRES,
    OneLevelSchwarz,
    MultiplicativeOneLevelSchwarz,
    py_demean as _py_demean,
    py_solve as _py_solve,
)


@dataclass(frozen=True)
class SolveResult:
    x: NDArray[np.float64]
    converged: Union[bool, NDArray[np.bool_]]
    iterations: Union[int, NDArray[np.intp]]
    residual: Union[float, NDArray[np.float64]]
    time_total: float
    time_setup: float
    time_solve: Union[float, NDArray[np.float64]]


@dataclass(frozen=True)
class DemeanResult:
    y_demean: NDArray[np.float64]
    converged: Union[bool, NDArray[np.bool_]]
    iterations: Union[int, NDArray[np.intp]]
    residual: Union[float, NDArray[np.float64]]
    time_total: float
    time_setup: float
    time_solve: Union[float, NDArray[np.float64]]


def _normalize_categories(categories):

    if isinstance(categories, list):
        categories = np.column_stack(categories).astype(np.uintp)
    return categories


def _wrap_solve_result(result: _PySolveResult, squeeze_output: bool) -> SolveResult:

    x = np.asarray(result.x)
    converged = np.asarray(result.converged)
    iterations = np.asarray(result.iterations)
    residual = np.asarray(result.residual)
    time_solve = np.asarray(result.time_solve)

    if squeeze_output:
        x = x[:, 0]
        converged = bool(converged[0])
        iterations = int(iterations[0])
        residual = float(residual[0])
        time_solve = float(time_solve[0])

    return SolveResult(
        x=x,
        converged=converged,
        iterations=iterations,
        residual=residual,
        time_total=float(result.time_total),
        time_setup=float(result.time_setup),
        time_solve=time_solve,
    )


def _wrap_demean_result(result: _PyDemeanResult, squeeze_output: bool) -> DemeanResult:

    y_demean = np.asarray(result.y_demean)
    converged = np.asarray(result.converged)
    iterations = np.asarray(result.iterations)
    residual = np.asarray(result.residual)
    time_solve = np.asarray(result.time_solve)

    if squeeze_output:
        y_demean = y_demean[:, 0]
        converged = bool(converged[0])
        iterations = int(iterations[0])
        residual = float(residual[0])
        time_solve = float(time_solve[0])

    return DemeanResult(
        y_demean=y_demean,
        converged=converged,
        iterations=iterations,
        residual=residual,
        time_total=float(result.time_total),
        time_setup=float(result.time_setup),
        time_solve=time_solve,
    )


def solve(categories, y, config=None, *, n_levels=None, weights=None, layout=None):
    """Solve fixed-effects normal equations.

    categories can be a 2D uintp array (n_obs, n_factors) or a list of 1D
    int64 arrays (one per factor).
    """
    if config is None:
        config = CG()

    categories = _normalize_categories(categories)
    y = np.asarray(y, dtype=np.float64)
    squeeze_output = y.ndim == 1
    if squeeze_output:
        y = y[:, None]
    elif y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array")
    result = _py_solve(
        categories, y, config, n_levels=n_levels, weights=weights, layout=layout
    )
    return _wrap_solve_result(result, squeeze_output)


def demean(categories, y, config=None, *, n_levels=None, weights=None, layout=None):
    """Demean y by removing fitted fixed effects using the same solver API."""
    if config is None:
        config = CG()

    categories = _normalize_categories(categories)
    y = np.asarray(y, dtype=np.float64)
    squeeze_output = y.ndim == 1
    if squeeze_output:
        y = y[:, None]
    elif y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array")
    result = _py_demean(
        categories, y, config, n_levels=n_levels, weights=weights, layout=layout
    )
    return _wrap_demean_result(result, squeeze_output)


__all__ = [
    "ApproxCholConfig",
    "ApproxSchurConfig",
    "DemeanResult",
    "SolveResult",
    "LSMR",
    "CG",
    "GMRES",
    "OneLevelSchwarz",
    "MultiplicativeOneLevelSchwarz",
    "demean",
    "solve",
]
