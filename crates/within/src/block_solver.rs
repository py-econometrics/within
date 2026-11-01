//! Schwarz local solver over approx-chol's irreducible-block factors.

use approx_chol::{BlockFactor, InnerParallelism};
use schwarz_precond::{LocalSolveError, LocalSolver};
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
}

impl LocalSolver for BlockSolver {
    fn n_local(&self) -> usize {
        self.blocks.iter().map(|b| b.vertices().len()).sum()
    }

    fn workspace_len(&self) -> usize {
        self.blocks
            .iter()
            .map(BlockFactor::scratch_len)
            .max()
            .unwrap_or(0)
    }

    fn solve_local(
        &self,
        x: &mut [f64],
        workspace: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let inner = if allow_inner_parallelism {
            InnerParallelism::Allow
        } else {
            InnerParallelism::Forbid
        };
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
