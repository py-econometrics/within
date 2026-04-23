//! Linear algebra layer: operator representations and preconditioner wiring.
//!
//! This module is the hub between the [`domain`](crate::domain) layer (which
//! builds subdomains from panel data) and the [`orchestrate`](crate::orchestrate)
//! layer (the public solve API). It provides the matrices and preconditioners
//! that power the iterative Krylov solves.
//!
//! # Operator representations
//!
//! Fixed-effects estimation reduces to solving the normal equations `G x = b`
//! where `G = D^T W D` is the Gramian of the weighted design matrix. This
//! module offers three representations of the matrices involved:
//!
//! | Representation | Type | Description |
//! |---|---|---|
//! | **D** (design matrix) | [`DesignOperator`] | Rectangular, implements `D x` and `D^T x` via gather/scatter on the observation store |
//! | **G implicit** | [`gramian::GramianOperator`] | Matrix-free `D^T W D x` — computes each matvec in three steps without storing G |
//! | **G explicit** | [`gramian::Gramian`] | Pre-assembled CSR sparse matrix — one-time build cost, then O(nnz) matvecs |
//!
//! Why multiple representations? **Memory vs speed tradeoff.** The implicit
//! Gramian avoids allocating the (potentially large) `G` matrix but requires
//! more FLOPs per matrix-vector product — it must touch every observation
//! twice per matvec. The explicit Gramian builds `G` once and reuses it,
//! giving cheaper matvecs at the cost of O(nnz(G)) memory. For problems
//! where `G` is much smaller than the observation data, explicit wins; for
//! very large or dense cross-tabulations, implicit may be preferable.
//!
//! # Submodules
//!
//! - [`gramian`] — Explicit `G = D^T W D` construction (CSR), cross-tabulation,
//!   and the implicit `GramianOperator`
//! - [`schwarz`] — Schwarz preconditioner construction: bridges fixed-effects
//!   types to the generic `schwarz-precond` API
//! - `local_solver` — Local subdomain solvers: approximate Cholesky (SDDM)
//!   and block-elimination backends
//! - `schur_complement` — Exact and approximate Schur complement computation
//!   for block-elimination local solves
//! - [`preconditioner`] — [`FePreconditioner`](preconditioner::FePreconditioner)
//!   enum dispatch over additive and multiplicative Schwarz
//! - `residual_update` — Residual update strategies for multiplicative Schwarz
//!   (observation-space vs sparse Gramian)
//! - `csr_block` — Internal rectangular CSR block used in bipartite Gramian
//!   structures

pub(crate) mod csr_block;
pub mod gramian;
pub(crate) mod local_solver;
pub mod preconditioner;
pub(crate) mod residual_update;
pub(crate) mod schur_complement;
pub mod schwarz;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// DesignOperator — rectangular, D·x / D^T·x (no weights)
// ---------------------------------------------------------------------------

use schwarz_precond::Operator;

use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

/// Operator wrapper around `&WeightedDesign<S>`.
///
/// `apply` = D·x (gather-add), `apply_adjoint` = D^T·x (scatter-add).
/// This is the raw design matrix — no observation weights.
pub struct DesignOperator<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
}

impl<'a, S: ObservationStore> DesignOperator<'a, S> {
    /// Wrap a weighted design matrix as a linear operator.
    pub fn new(design: &'a WeightedDesign<S>) -> Self {
        Self { design }
    }
}

impl<S: ObservationStore> Operator for DesignOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_rows
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.design.matvec_d(x, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.design.rmatvec_dt(x, y);
    }
}

// ---------------------------------------------------------------------------
// WeightedDesignOperator — rectangular, W^{1/2}·D·x / D^T·W^{1/2}·x
// ---------------------------------------------------------------------------

use std::sync::Mutex;

/// Weighted rectangular design operator: `A = W^{1/2} D`.
///
/// `apply` = `W^{1/2} D x` (observation space), `apply_adjoint` = `D^T W^{1/2} x` (DOF space).
/// For unweighted designs, delegates directly to `D x` / `D^T x` with no extra work.
///
/// The normal equations of this operator give `A^T A = D^T W D = G` (the Gramian),
/// so the existing Schwarz preconditioner approximating `G^{-1}` can be used directly.
pub struct WeightedDesignOperator<'a, S: ObservationStore> {
    design: &'a WeightedDesign<S>,
    /// Pre-computed `sqrt(w_i)` per observation. `None` when unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// Scratch for the adjoint path: stores `sqrt(w_i) * u_i`.
    scratch: Mutex<Vec<f64>>,
}

impl<'a, S: ObservationStore> WeightedDesignOperator<'a, S> {
    /// Create from a weighted design matrix.
    pub fn new(design: &'a WeightedDesign<S>) -> Self {
        let sqrt_weights = if design.store.is_unweighted() {
            None
        } else {
            Some(
                (0..design.n_rows)
                    .map(|i| design.uid_weight(i).sqrt())
                    .collect(),
            )
        };
        Self {
            scratch: Mutex::new(vec![0.0; design.n_rows]),
            design,
            sqrt_weights,
        }
    }

    /// Compute the observation-space RHS `b = W^{1/2} y`.
    ///
    /// For unweighted designs, returns a copy of `y`.
    pub fn weighted_rhs(&self, y: &[f64]) -> Vec<f64> {
        match &self.sqrt_weights {
            None => y.to_vec(),
            Some(sw) => y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect(),
        }
    }
}

impl<S: ObservationStore> Operator for WeightedDesignOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_rows
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        // y = W^{1/2} (D x)
        self.design.matvec_d(x, y);
        if let Some(sw) = &self.sqrt_weights {
            for (yi, &swi) in y.iter_mut().zip(sw) {
                *yi *= swi;
            }
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        // y = D^T (W^{1/2} x)
        match &self.sqrt_weights {
            None => self.design.rmatvec_dt(x, y),
            Some(sw) => {
                let mut tmp = self.scratch.lock().unwrap();
                for (ti, (&xi, &swi)) in tmp.iter_mut().zip(x.iter().zip(sw)) {
                    *ti = swi * xi;
                }
                self.design.rmatvec_dt(&tmp, y);
            }
        }
    }
}
