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

    fn describe(&self) -> String {
        let mut d: Vec<String> = self.blocks.iter().map(|b| b.proto_describe()).collect();
        d.sort_by_key(|s| std::cmp::Reverse(s.len()));
        format!("{} blocks; {}", self.blocks.len(), d.into_iter().take(3).collect::<Vec<_>>().join(" | "))
    }

    fn solve_local_rows<const K: usize>(&self, x: &mut [[f64; K]], threaded: bool) {
        thread_local! {
            static WORK: std::cell::Cell<(Vec<f64>, Vec<f64>, Vec<f64>)> = const { std::cell::Cell::new((Vec::new(), Vec::new(), Vec::new())) };
        }
        // Taken, not borrowed: nested rayon work may run another solve on this thread.
        let (mut work, mut scratch, mut col) = WORK.take();
        scratch.resize(scratch.len().max(self.scratch_size), 0.0);
        col.resize(col.len().max(self.n_local), 0.0);
        let mut off = 0;
        for block in &self.blocks {
            let nb = block.vertices().len();
            block.proto_solve_rows::<K>(&mut x[off..off + nb], &mut work, &mut scratch, &mut col, threaded);
            off += nb;
        }
        WORK.set((work, scratch, col));
    }

    fn solve_local_block(&self, x: &mut [f64], k: usize, threaded: bool) {
        let n = self.n_local;
        let mut scratch = vec![0.0; self.scratch_size];
        let mut buf = vec![0.0; n * 16];
        let mut col = vec![0.0; n];
        let mut first = 0;
        while first < k {
            let group = [16, 8, 4, 2, 1].into_iter().find(|&g| g <= k - first).unwrap();
            let mut off = 0;
            for block in &self.blocks {
                let nb = block.vertices().len();
                let xs = &mut x[first * n + off..];
                let (s, b, c) = (&mut scratch[..block.scratch_len()], &mut buf[..], &mut col[..]);
                match group {
                    16 => block.proto_solve_k::<16>(xs, n, b, s, c, threaded),
                    8 => block.proto_solve_k::<8>(xs, n, b, s, c, threaded),
                    4 => block.proto_solve_k::<4>(xs, n, b, s, c, threaded),
                    2 => block.proto_solve_k::<2>(xs, n, b, s, c, threaded),
                    _ => block.proto_solve_k::<1>(xs, n, b, s, c, threaded),
                }
                off += nb;
            }
            first += group;
        }
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.blocks.iter().map(BlockFactor::splittable_work).sum()
    }
}
