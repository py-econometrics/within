"""Type stubs for the Rust extension module."""

import numpy as np
from numpy.typing import NDArray

class ApproxCholConfig:
    seed: int
    split: int
    def __init__(
        self,
        seed: int = 0,
        split: int = 1,
    ) -> None: ...

class ApproxSchurConfig:
    seed: int
    def __init__(self, seed: int = 0) -> None: ...

class OneLevelSchwarz:
    smoother: ApproxCholConfig | None
    local_solver: str | None
    approx_schur: ApproxSchurConfig | None
    dense_schur_threshold: int | None
    def __init__(
        self,
        smoother: ApproxCholConfig | None = None,
        local_solver: str | None = None,
        approx_schur: ApproxSchurConfig | None = None,
        dense_schur_threshold: int | None = None,
    ) -> None: ...

class MultiplicativeOneLevelSchwarz:
    """One-level multiplicative Schwarz with sequential subdomain sweeps."""
    smoother: ApproxCholConfig | None
    local_solver: str | None
    approx_schur: ApproxSchurConfig | None
    dense_schur_threshold: int | None
    def __init__(
        self,
        smoother: ApproxCholConfig | None = None,
        local_solver: str | None = None,
        approx_schur: ApproxSchurConfig | None = None,
        dense_schur_threshold: int | None = None,
    ) -> None: ...

class LSMR:
    tol: float
    maxiter: int
    conlim: float
    def __init__(self, tol: float = 1e-8, maxiter: int = 1000, conlim: float = 1e8) -> None: ...

class CG:
    tol: float
    maxiter: int
    preconditioner: OneLevelSchwarz | MultiplicativeOneLevelSchwarz | None
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        preconditioner: OneLevelSchwarz | MultiplicativeOneLevelSchwarz | None = None,
    ) -> None: ...

class GMRES:
    tol: float
    maxiter: int
    restart: int
    preconditioner: MultiplicativeOneLevelSchwarz
    def __init__(
        self, preconditioner: MultiplicativeOneLevelSchwarz,
        tol: float = 1e-8,
        maxiter: int = 10000,
        restart: int = 30,
    ) -> None: ...

class SolveResult:
    """Result of a solve operation."""

    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def time_total(self) -> float: ...
    @property
    def time_setup(self) -> float: ...
    @property
    def time_solve(self) -> float: ...

def py_solve(
    categories: NDArray[np.uintp],
    n_levels: list[int],
    y: NDArray[np.float64],
    config: LSMR | CG | GMRES,
    weights: NDArray[np.float64] | None = None,
    layout: str | None = None,
) -> SolveResult: ...

def py_generate_synthetic_data(
    n_levels: list[int],
    n_rows: int,
    seed: int | None = None,
) -> tuple[NDArray[np.uintp], NDArray[np.float64], NDArray[np.float64]]: ...
