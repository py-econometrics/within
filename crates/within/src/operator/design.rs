use std::borrow::Cow;

use portable_atomic::AtomicF64;
use schwarz_precond::Operator;

use crate::domain::{Design, TermMeta};

mod gather;
mod scatter;

pub(crate) use gather::gather_apply;
use scatter::scatter_apply;

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// A [`TermMeta`] with its frame columns resolved into borrows for the kernels.
struct ResolvedTerm<'a> {
    meta: &'a TermMeta,
    levels: &'a [u32],
    zs: Vec<&'a [f64]>,
}

fn resolve_terms<'a>(design: &'a Design<'_>) -> Vec<ResolvedTerm<'a>> {
    design
        .terms
        .iter()
        .enumerate()
        .map(|(q, t)| ResolvedTerm {
            meta: t,
            levels: design.frame.level_column(q),
            zs: t
                .slopes
                .iter()
                .map(|&c| design.frame.loading_column(c))
                .collect(),
        })
        .collect()
}

// ===========================================================================
// DesignOperator — D, optionally rescaled by W^{1/2}
// ===========================================================================

/// Rectangular design operator: `D` (unweighted) or `W^{1/2} D` (weighted).
///
/// `apply` = `D x` / `W^{1/2} D x` (gather), `apply_adjoint` = `D^T x` /
/// `D^T W^{1/2} x` (scatter). For the weighted variant, the normal equations
/// `A^T A = D^T W D = G` recover the Gramian, so the same Schwarz
/// preconditioner approximating `G^{-1}` applies. Pass `None` to
/// [`DesignOperator::new`] for `D`, or `Some(&w)` for `W^{1/2} D`. The branch
/// on weights is hoisted outside the per-row loop — the weighted gather applies
/// `W^{1/2}` in a trailing per-chunk sweep, and the adjoint multiplies inline
/// through a closure, so there is no per-row scratch buffer.
pub(crate) struct DesignOperator<'a> {
    design: &'a Design<'a>,
    sqrt_weights: Option<Vec<f64>>,
    /// Reusable atomic-scatter scratch, sized once to the largest term's
    /// coefficient block and reused across terms and `apply_adjoint` calls so
    /// it is allocated once per operator rather than once per LSMR iteration.
    /// `apply_adjoint` takes `&self`, but `AtomicF64`'s load/store/fetch_add are
    /// `&self` operations, so a plain `Vec<AtomicF64>` (already `Sync`) needs no
    /// lock: each call re-seeds the buffer via `store` instead of resizing it.
    scatter_scratch: Vec<AtomicF64>,
}

impl<'a> DesignOperator<'a> {
    /// Wrap a design matrix as a linear operator.
    ///
    /// Pass `None` for `D`, `Some(&w)` for `W^{1/2} D` (then `w.len()` must
    /// equal `design.n_obs`). Precomputes and stores `sqrt(W)` when weights
    /// are present.
    ///
    /// # Panics
    ///
    /// Panics when `weights.is_some()` and `weights.unwrap().len()` does not
    /// equal `design.n_obs`. The `Solver` entry points perform fallible
    /// validation against `BuildError::WeightCountMismatch` before
    /// construction, so callers that go through `Solver::new` or
    /// `solve()` never trigger this panic.
    pub(crate) fn new(design: &'a Design<'a>, weights: Option<&[f64]>) -> Self {
        let sqrt_weights = weights.map(|w| {
            assert_eq!(
                w.len(),
                design.n_obs,
                "weights length {} does not match design.n_obs {}",
                w.len(),
                design.n_obs
            );
            w.iter().map(|wi| wi.sqrt()).collect()
        });
        let max_block = design.terms.iter().map(|t| t.n_dofs()).max().unwrap_or(0);
        Self {
            design,
            sqrt_weights,
            scatter_scratch: (0..max_block).map(|_| AtomicF64::new(0.0)).collect(),
        }
    }

    /// Compute the observation-space RHS `b = W^{1/2} y`.
    ///
    /// For unweighted designs, borrows `y` (no allocation); the weighted
    /// variant returns an owned scaled copy.
    pub(crate) fn weighted_rhs<'y>(&self, y: &'y [f64]) -> Cow<'y, [f64]> {
        match &self.sqrt_weights {
            None => Cow::Borrowed(y),
            Some(sw) => Cow::Owned(y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect()),
        }
    }
}

impl Operator for DesignOperator<'_> {
    fn nrows(&self) -> usize {
        self.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        gather_apply(self.design, x, y, self.sqrt_weights.as_deref());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_obs);
        debug_assert_eq!(y.len(), self.design.n_dofs);
        y.fill(0.0);
        // Reuse the per-operator atomic-scatter scratch across iterations. No
        // lock needed: `AtomicF64` mutates through `&self`, and a single
        // operator's `apply_adjoint` calls are sequential (`solve_batch` builds
        // a distinct operator per RHS), so the shared buffer is never raced.
        match &self.sqrt_weights {
            Some(sw) => scatter_apply(self.design, &self.scatter_scratch, y, &|i| sw[i] * x[i]),
            None => scatter_apply(self.design, &self.scatter_scratch, y, &|i| x[i]),
        }
        Ok(())
    }
}
