from within._within import (
    OperatorRepr,
    Preconditioner,
    SolveResult,
    BatchSolveResult,
    CG,
    GMRES,
    FePreconditioner,
    Solver,
    solve,
    solve_batch,
    AdditiveSchwarz,
    MultiplicativeSchwarz,
)

# Convenience aliases
Additive = AdditiveSchwarz
Multiplicative = MultiplicativeSchwarz

from within.utils import make_akm_panel

__all__ = [
    "OperatorRepr",
    "Preconditioner",
    "SolveResult",
    "BatchSolveResult",
    "CG",
    "GMRES",
    "FePreconditioner",
    "Solver",
    "solve",
    "solve_batch",
    "Additive",
    "Multiplicative",
    "AdditiveSchwarz",
    "MultiplicativeSchwarz",
    "make_akm_panel",
]