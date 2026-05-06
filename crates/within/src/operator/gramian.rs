//! Gramian `G = D^T W D` — explicit construction and implicit operators.
//!
//! # What is the Gramian?
//!
//! In fixed-effects estimation, the design matrix `D` maps DOF coefficients to
//! observations. The normal equations for the weighted least-squares problem
//! `min_x || W^{1/2} (y - D x) ||^2` are `G x = D^T W y`, where the Gramian
//! `G = D^T W D` is the system matrix. This module builds `G` and provides
//! operators to apply it.
//!
//! # CSR layout and block structure
//!
//! The explicit [`Gramian`] stores `G` as a CSR (Compressed Sparse Row)
//! matrix via [`SparseMatrix`]. Since the Gramian is
//! symmetric, CSR and CSC are equivalent; CSR is the standard choice.
//!
//! The matrix has a natural block structure reflecting the factor grouping:
//!
//! - **Diagonal blocks** (`G_{qq}`) — weighted counts within each factor
//!   level.
//! - **Off-diagonal blocks** (`G_{qr}`) — cross-tabulations between factor
//!   pairs.
//!
//! # Explicit vs implicit
//!
//! - **Explicit** ([`Gramian`]): pre-assembles the CSR matrix once; subsequent
//!   matvecs cost O(nnz(G)).
//! - **Implicit** ([`GramianOperator`]): computes `G x` on the fly via
//!   `D^T (W (D x))`. Construct with `GramianOperator::new(design, weights)`,
//!   passing `None` for `D^T D` or `Some(&w)` for `D^T W D`. The
//!   weighted/unweighted choice branches once per matvec — the per-row scatter
//!   loop stays branch-free, monomorphized over distinct closure types.

mod cross_tab;
mod explicit;
#[cfg(test)]
mod tests;

pub(crate) use cross_tab::{find_all_active_levels, BipartiteComponent, CrossTab};

use std::sync::Arc;
use std::sync::Mutex;

use schwarz_precond::{Operator, SparseMatrix};

use super::{gather_apply, scatter_apply};
use crate::domain::Design;
use crate::observation::Store;

/// Explicit Gramian `G = D^T W D` stored as CSR.
pub struct Gramian {
    /// The assembled CSR sparse matrix `G = D^T W D`.
    pub matrix: Arc<SparseMatrix>,
}

/// Implicit Gramian operator: computes `D^T (W (D x))`.
///
/// Weights are optional. The unweighted case (`D^T D`) and weighted case
/// (`D^T W D`) share one struct; the per-matvec branch is hoisted outside the
/// scatter loop, so each variant compiles to its own monomorphized hot kernel.
pub struct GramianOperator<'a, S: Store> {
    design: &'a Design<S>,
    weights: Option<&'a [f64]>,
    scratch: Mutex<Vec<f64>>,
}

impl<'a, S: Store> GramianOperator<'a, S> {
    /// Create an implicit Gramian operator. Pass `None` for `D^T D`,
    /// `Some(&w)` for `D^T W D` (then `w.len()` must equal `design.n_rows`).
    pub fn new(design: &'a Design<S>, weights: Option<&'a [f64]>) -> Self {
        if let Some(w) = weights {
            assert_eq!(
                w.len(),
                design.n_rows,
                "weights length {} does not match design.n_rows {}",
                w.len(),
                design.n_rows
            );
        }
        Self {
            scratch: Mutex::new(vec![0.0; design.n_rows]),
            design,
            weights,
        }
    }
}

impl<S: Store> Operator for GramianOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_dofs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let mut tmp = self.scratch.lock().unwrap();
        gather_apply(self.design, x, &mut tmp, |_, s| s); // tmp = D x
        y.fill(0.0);
        match self.weights {
            Some(w) => scatter_apply(self.design, y, |i| w[i] * tmp[i]), // y = D^T W tmp
            None => scatter_apply(self.design, y, |i| tmp[i]),           // y = D^T tmp
        }
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.apply(x, y)
    }
}
