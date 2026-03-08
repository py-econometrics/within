from within._within import (
    OperatorRepr,
    Preconditioner,
    SolveResult,
    BatchSolveResult,
    CG,
    GMRES,
    Solver,
    solve,
    solve_batch,
    AdditiveSchwarz,
    MultiplicativeSchwarz,
)

# Convenience aliases
Additive = AdditiveSchwarz
Multiplicative = MultiplicativeSchwarz

__all__ = [
    "OperatorRepr",
    "Preconditioner",
    "SolveResult",
    "BatchSolveResult",
    "CG",
    "GMRES",
    "Solver",
    "solve",
    "solve_batch",
    "Additive",
    "Multiplicative",
    "AdditiveSchwarz",
    "MultiplicativeSchwarz",
]
