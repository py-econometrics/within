//! Schwarz local solver over approx-chol's irreducible-block factors.

use approx_chol::{BlockFactor, InnerParallelism};
use schwarz_precond::{LocalSolver, MatMut, MemStack, Par, StackReq};
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

    /// Whether a solve has enough splittable work to pay for nested rayon.
    fn splits(&self) -> bool {
        const MIN_INNER_PARALLEL_WORK: usize = 200_000;
        let work = self
            .blocks
            .iter()
            .map(BlockFactor::splittable_work)
            .sum::<usize>();
        work >= MIN_INNER_PARALLEL_WORK
    }

    fn scratch_len(&self) -> usize {
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

    fn solve_local_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::new::<f64>(self.scratch_len())
    }

    fn solve_local(&self, x: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        let inner = if par.degree() > 1 && self.splits() {
            InnerParallelism::Allow
        } else {
            InnerParallelism::Forbid
        };
        let (mut scratch, _) = stack.make_with(self.scratch_len(), |_| 0.0);
        for column in x.col_iter_mut() {
            let mut rest = column
                .try_as_col_major_mut()
                .expect("Schwarz passes unit-stride columns")
                .as_slice_mut();
            for block in &self.blocks {
                let (local, tail) = rest.split_at_mut(block.vertices().len());
                block
                    .solve_local(local, &mut scratch[..block.scratch_len()], inner)
                    .expect("buffers sized by the factor");
                rest = tail;
            }
        }
    }
}
