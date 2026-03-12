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
    ReductionStrategy,
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
    "FePreconditioner",
    "Solver",
    "solve",
    "solve_batch",
    "Additive",
    "Multiplicative",
    "AdditiveSchwarz",
    "MultiplicativeSchwarz",
    "ReductionStrategy",
]
