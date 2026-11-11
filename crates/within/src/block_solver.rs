//! Schwarz local solver over approx-chol's block-diagonal approximate inverse of one channel pair.

use approx_chol::ApproxInverse;
use schwarz_precond::{LocalSolveError, LocalSolver, MemStack, StackReq, Threads};
use serde::{Deserialize, Serialize};

/// One pair's factor; the subdomain's local order is the factored matrix's numbering.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct BlockSolver {
    inverse: ApproxInverse,
}

impl BlockSolver {
    pub(crate) fn new(inverse: ApproxInverse) -> Self {
        Self { inverse }
    }
}

fn budget(threads: Threads) -> approx_chol::Threads {
    match threads {
        Threads::All => approx_chol::Threads::All,
        Threads::Count(n) => approx_chol::Threads::Count(n),
    }
}

impl LocalSolver for BlockSolver {
    fn n_local(&self) -> usize {
        self.inverse.n()
    }

    fn scratch(&self, threads: Threads) -> StackReq {
        self.inverse.scratch(1, budget(threads))
    }

    fn solve_local(
        &self,
        x: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), LocalSolveError> {
        self.inverse
            .apply_with(x, budget(threads), stack)
            .map_err(|e| LocalSolveError::BackendFailed {
                context: "BlockSolver::solve_local",
                message: e.to_string(),
            })
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.inverse.n_steps()
    }
}
