//! Schwarz local solver over approx-chol's factors of a channel pair's irreducible blocks.

use approx_chol::{BlockFactor, InnerParallelism};
use schwarz_precond::{LocalSolveError, LocalSolver};

/// Irreducible blocks of one channel pair, solved one after another in local order.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "Blocks")]
pub(crate) struct BlockSolver {
    blocks: Vec<BlockFactor>,
    #[serde(skip)]
    n_local: usize,
    #[serde(skip)]
    scratch_size: usize,
}

/// The wire form: everything else is derived from the blocks.
#[derive(serde::Deserialize)]
struct Blocks {
    blocks: Vec<BlockFactor>,
}

impl From<Blocks> for BlockSolver {
    fn from(Blocks { blocks }: Blocks) -> Self {
        Self::new(blocks)
    }
}

impl BlockSolver {
    pub(crate) fn new(blocks: Vec<BlockFactor>) -> Self {
        let n_local = blocks.iter().map(|block| block.vertices().len()).sum();
        let scratch = blocks.iter().map(BlockFactor::scratch_len).max();
        Self {
            blocks,
            n_local,
            scratch_size: scratch.unwrap_or(0).max(n_local),
        }
    }

    #[cfg(test)]
    pub(crate) fn blocks_for_test(&self) -> &[BlockFactor] {
        &self.blocks
    }

    /// Local order, in the pair's `[rows | cols]` numbering.
    pub(crate) fn vertices(&self) -> impl Iterator<Item = usize> + '_ {
        self.blocks.iter().flat_map(BlockFactor::vertices)
    }
}

impl LocalSolver for BlockSolver {
    fn n_local(&self) -> usize {
        self.n_local
    }

    fn scratch_size(&self) -> usize {
        self.scratch_size
    }

    /// Solves in `rhs` with `sol` as the blocks' workspace, then copies the solution over.
    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        let inner = match allow_inner_parallelism {
            true => InnerParallelism::Allow,
            false => InnerParallelism::Forbid,
        };
        let mut rest = &mut rhs[..self.n_local];
        for block in &self.blocks {
            let (x, tail) = rest.split_at_mut(block.vertices().len());
            block
                .solve_local(x, &mut sol[..block.scratch_len()], inner)
                .map_err(|e| LocalSolveError::BackendFailed {
                    context: "within.local.block_factor",
                    message: e.to_string(),
                })?;
            rest = tail;
        }
        sol[..self.n_local].copy_from_slice(&rhs[..self.n_local]);
        Ok(())
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.blocks.iter().map(BlockFactor::splittable_work).sum()
    }
}
