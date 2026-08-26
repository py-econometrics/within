"""High-performance fixed-effects solver for econometric panel data.

``within`` solves the normal equations arising from multi-way fixed-effect
models (D^T W D x = D^T W y) using modified LSMR with a domain-decomposition
(Schwarz) preconditioner. The heavy lifting is done in Rust; this package
provides the Python API.

Quick start::

    import numpy as np
    import within

    # Two factors, four observations
    categories = np.array(
        [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]],
        dtype=np.uint32,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])

    result = within.solve(categories, y)
    print(result.x)         # fixed-effect coefficients
    print(result.converged) # True

For repeated solves on the same panel structure, use the persistent
``Design`` and ``Solver`` classes to amortise design construction and
preconditioner setup::

    design = within.Design(categories)
    solver = within.Solver(design)
    r1 = solver.solve(y1)
    r2 = solver.solve(y2)

Two-tier public API:

- The top-level ``within`` namespace exposes the call-site essentials
  (``solve``, ``Design``, ``Solver``, ``LsmrOptions``, ``PreconditionerConfig``,
  ``Preconditioner``).
- :mod:`within.config` re-exports the advanced configuration objects
  (``LocalSolverConfig``, ``ApproxCholConfig``, ``ApproxSchurConfig``,
  ``ReductionStrategy``, ``ScalingConfig``, ``Schur``).

Thread safety: the heavy work runs with the interpreter detached and reads the
caller's arrays in place. As with NumPy itself, an input array must not be
mutated from another thread while a ``within`` call is reading it.

For Rust-level internals, build the API docs with ``cargo doc --open``.
"""

from within import config  # noqa: F401 — expose submodule on `within.config`
from within._within import (
    BatchSolveResult,
    CoefficientLayout,
    Design,
    Effect,
    LsmrOptions,
    Preconditioner,
    PreconditionerConfig,
    Solver,
    SolveResult,
    UnidentifiedDirection,
    solve,
    solve_batch,
)

__all__ = [
    "BatchSolveResult",
    "CoefficientLayout",
    "Effect",
    "LsmrOptions",
    "Preconditioner",
    "PreconditionerConfig",
    "SolveResult",
    "Solver",
    "Design",
    "UnidentifiedDirection",
    "solve",
    "solve_batch",
    "config",
]
