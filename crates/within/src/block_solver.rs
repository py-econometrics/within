//! Schwarz local solver over approx-chol's irreducible-block factors.

use approx_chol::{BlockFactor, InnerParallelism};
use schwarz_precond::{LocalSolveError, LocalSolver, MemStack, StackReq, Threads};
use serde::{Deserialize, Serialize};

/// Consecutive irreducible blocks; the subdomain's local order concatenates their vertices.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct BlockSolver {
    blocks: Vec<BlockFactor>,
}

impl BlockSolver {
    pub(crate) fn new(blocks: Vec<BlockFactor>) -> Self {
        Self { blocks }
    }

    fn workspace_len(&self) -> usize {
        self.blocks
            .iter()
            .map(BlockFactor::scratch_len)
            .max()
            .unwrap_or(0)
    }
}

impl LocalSolver for BlockSolver {
    fn n_local(&self) -> usize {
        self.blocks.iter().map(|b| b.vertices().len()).sum()
    }

    fn scratch(&self, _threads: Threads) -> StackReq {
        StackReq::new::<f64>(self.workspace_len())
    }

    fn solve_local(
        &self,
        x: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), LocalSolveError> {
        let inner = match threads {
            Threads::All => InnerParallelism::Allow,
            Threads::Count(_) => InnerParallelism::Forbid,
        };
        let (mut workspace, _) = stack.make_with(self.workspace_len(), |_| 0.0);
        let mut rest = x;
        for block in &self.blocks {
            let (local, tail) = rest.split_at_mut(block.vertices().len());
            block
                .solve_local(local, &mut workspace[..block.scratch_len()], inner)
                .map_err(|e| LocalSolveError::BackendFailed {
                    context: "BlockSolver::solve_local",
                    message: e.to_string(),
                })?;
            rest = tail;
        }
        Ok(())
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.blocks.iter().map(BlockFactor::splittable_work).sum()
    }
}
