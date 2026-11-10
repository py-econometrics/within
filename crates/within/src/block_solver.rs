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
        let n_local = self.n_local();
        scratch.resize(scratch.len().max(self.workspace_len()), 0.0);
        col.resize(col.len().max(n_local), 0.0);
        let mut off = 0;
        for block in &self.blocks {
            let nb = block.vertices().len();
            block.proto_solve_rows::<K>(&mut x[off..off + nb], &mut work, &mut scratch, &mut col, threaded);
            off += nb;
        }
        WORK.set((work, scratch, col));
    }

    fn solve_local_block(&self, x: &mut [f64], k: usize, threaded: bool) {
        let n = self.n_local();
        let mut scratch = vec![0.0; self.workspace_len()];
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
