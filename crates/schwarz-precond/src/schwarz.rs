//! Schwarz preconditioners: additive and multiplicative variants.

use std::cell::Cell;

mod additive;
mod multiplicative;

pub use additive::{AdditiveSchwarzDiagnostics, ReductionStrategy, SchwarzPreconditioner};
pub use multiplicative::{
    MultiplicativeSchwarzPreconditioner, OperatorResidualUpdater, ResidualUpdater,
};

std::thread_local! {
    static LOCAL_SOLVER_INNER_PARALLELISM: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether local solvers may spawn nested Rayon work on this thread.
pub fn local_solver_inner_parallelism_enabled() -> bool {
    LOCAL_SOLVER_INNER_PARALLELISM.with(Cell::get)
}

/// Runs `f` with the local-solver nested-parallelism flag set to `enabled`.
pub fn with_local_solver_inner_parallelism<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    LOCAL_SOLVER_INNER_PARALLELISM.with(|flag| {
        let previous = flag.replace(enabled);
        let result = f();
        flag.set(previous);
        result
    })
}
