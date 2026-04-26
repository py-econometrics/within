//! Gramian `G = D^T W D` — explicit construction and implicit operator.
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
//!   level. For an unweighted design, `G_{qq}[i,i]` is simply the number
//!   of observations at level `i` of factor `q`.
//! - **Off-diagonal blocks** (`G_{qr}`) — cross-tabulations between factor
//!   pairs. Entry `G_{qr}[i,j]` is the (weighted) count of observations
//!   that belong to level `i` of factor `q` *and* level `j` of factor `r`.
//!
//! # Explicit vs implicit
//!
//! - **Explicit** ([`Gramian`]): pre-assembles the CSR matrix once. Each
//!   subsequent matvec costs O(nnz(G)). Best when the Gramian is reused
//!   many times (e.g., hundreds of Krylov iterations) and fits in memory.
//! - **Implicit** ([`GramianOperator`]): computes `G x` on the fly as
//!   `D^T(W(D x))` — three passes over the observation data, no extra
//!   storage. Better when G would be very large or when memory is tight.

mod cross_tab;
mod explicit;
#[cfg(test)]
mod tests;

pub(crate) use cross_tab::{find_all_active_levels, BipartiteComponent, CrossTab};

use std::sync::Arc;
use std::sync::Mutex;

use schwarz_precond::{Operator, SparseMatrix};

use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

/// Explicit Gramian `G = D^T W D` stored as CSR.
pub struct Gramian {
    /// The assembled CSR sparse matrix `G = D^T W D`.
    pub matrix: Arc<SparseMatrix>,
}

/// Implicit weighted Gramian operator: computes `D^T(W·(D·x))`.
///
/// For unweighted designs, this is `D^T(D·x)`.
/// Scratch space avoids per-call allocation.
pub struct GramianOperator<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
    scratch: Mutex<Vec<f64>>,
}

impl<'a, S: ObservationStore> GramianOperator<'a, S> {
    /// Create an implicit Gramian operator from a weighted design.
    pub fn new(design: &'a WeightedDesign<S>) -> Self {
        Self {
            scratch: Mutex::new(vec![0.0; design.n_rows]),
            design,
        }
    }
}

impl<S: ObservationStore> Operator for GramianOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_dofs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let mut tmp = self.scratch.lock().unwrap();
        tmp.fill(0.0);
        self.design.matvec_d(x, &mut tmp);
        self.design.rmatvec_wdt(&tmp, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.apply(x, y);
    }
}
