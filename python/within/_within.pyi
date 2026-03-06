"""Type stubs for the Rust extension module."""

import numpy as np
from numpy.typing import NDArray
from enum import IntEnum

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

class OperatorRepr(IntEnum):
    Implicit = 0
    Explicit = 1

class SchurComplement:
    approx_chol: ApproxCholConfig | None
    approx_schur: ApproxSchurConfig | None
    dense_threshold: int
    def __init__(
        self,
        approx_chol: ApproxCholConfig | None = None,
        approx_schur: ApproxSchurConfig | None = None,
        dense_threshold: int = 24,
    ) -> None: ...

class FullSddm:
    approx_chol: ApproxCholConfig | None
    def __init__(self, approx_chol: ApproxCholConfig | None = None) -> None: ...

class AdditiveSchwarz:
    local_solver: SchurComplement | FullSddm | None
    def __init__(
        self,
        local_solver: SchurComplement | FullSddm | None = None,
    ) -> None: ...

class MultiplicativeSchwarz:
    """One-level multiplicative Schwarz with sequential subdomain sweeps."""

    local_solver: SchurComplement | FullSddm | None
    def __init__(
        self,
        local_solver: SchurComplement | FullSddm | None = None,
    ) -> None: ...

class CG:
    tol: float
    maxiter: int
    preconditioner: AdditiveSchwarz | None
    operator: OperatorRepr
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        preconditioner: AdditiveSchwarz | None = None,
        operator: OperatorRepr = OperatorRepr.Implicit,
    ) -> None: ...

class GMRES:
    tol: float
    maxiter: int
    restart: int
    preconditioner: AdditiveSchwarz | MultiplicativeSchwarz | None
    operator: OperatorRepr
    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        restart: int = 30,
        preconditioner: AdditiveSchwarz | MultiplicativeSchwarz | None = None,
        operator: OperatorRepr = OperatorRepr.Implicit,
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

def solve(
    categories: NDArray[np.uintp],
    y: NDArray[np.float64],
    config: CG | GMRES | None = None,
    n_levels: list[int] | None = None,
    weights: NDArray[np.float64] | None = None,
) -> SolveResult: ...
