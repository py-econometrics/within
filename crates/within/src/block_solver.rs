//! Schwarz local solver over approx-chol's block-diagonal approximate inverse of one channel pair.

use std::cell::Cell;

use approx_chol::{ApproxInverse, ColumnMajor, MatrixMut, MemBuffer, MemStack, StackReq, Threads};
use schwarz_precond::{LocalSolveError, LocalSolver};
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

    fn apply<M: MatrixMut<f64> + ?Sized>(
        &self,
        x: &mut M,
        columns: usize,
        threaded: bool,
    ) -> Result<(), LocalSolveError> {
        let threads = if threaded {
            Threads::All
        } else {
            Threads::Count(std::num::NonZeroUsize::MIN)
        };
        let need = self.inverse.scratch(columns, threads);
        with_stack(need, |stack| self.inverse.apply_with(x, threads, stack)).map_err(|e| {
            LocalSolveError::BackendFailed {
                context: "BlockSolver::solve_local",
                message: e.to_string(),
            }
        })
    }
}

/// Runs `f` on a stack holding `need`, reusing this thread's buffer once it is large enough.
fn with_stack<R>(need: StackReq, f: impl FnOnce(&mut MemStack) -> R) -> R {
    thread_local! {
        static BUFFER: Cell<Option<(StackReq, MemBuffer)>> = const { Cell::new(None) };
    }
    // Taken, not borrowed: nested rayon work may run another solve on this thread.
    let (held, mut buffer) = match BUFFER.take() {
        Some((held, buffer)) if held.or(need) == held => (held, buffer),
        Some((held, _)) => (held.or(need), MemBuffer::new(held.or(need))),
        None => (need, MemBuffer::new(need)),
    };
    let result = f(MemStack::new(&mut buffer));
    BUFFER.set(Some((held, buffer)));
    result
}

impl LocalSolver for BlockSolver {
    fn n_local(&self) -> usize {
        self.inverse.n()
    }

    fn workspace_len(&self) -> usize {
        0
    }

    fn solve_local(
        &self,
        x: &mut [f64],
        _workspace: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        self.apply(x, 1, allow_inner_parallelism)
    }

    fn solve_local_rows<const K: usize>(&self, x: &mut [[f64; K]], threaded: bool) {
        self.apply(x, K, threaded).expect("local solve");
    }

    fn solve_local_block(&self, x: &mut [f64], k: usize, threaded: bool) {
        let n = self.n_local();
        let mut columns = ColumnMajor::new(&mut x[..n * k], n).expect("k columns of n rows");
        self.apply(&mut columns, k, threaded).expect("local solve");
    }

    fn inner_parallelism_work_estimate(&self) -> usize {
        self.inverse.n_steps()
    }
}
