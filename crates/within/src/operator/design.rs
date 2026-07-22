use std::borrow::Cow;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};

use portable_atomic::AtomicF64;
use schwarz_precond::Operator;

use crate::domain::Design;

mod gather;
mod scatter;

pub(crate) use gather::gather_apply;
use scatter::scatter_apply;

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

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
    /// Debug-only reentry sentinel: a concurrent `apply_adjoint` would race the
    /// `scatter_scratch` writes it makes through `&self`.
    #[cfg(debug_assertions)]
    adjoint_active: AtomicBool,
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
            #[cfg(debug_assertions)]
            adjoint_active: AtomicBool::new(false),
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

/// RAII reentry guard: `acquire` asserts no call is in flight; `Drop` clears
/// the flag on every exit path, panics included.
#[cfg(debug_assertions)]
struct ReentryGuard<'a>(&'a AtomicBool);

#[cfg(debug_assertions)]
impl<'a> ReentryGuard<'a> {
    fn acquire(active: &'a AtomicBool) -> Self {
        let already_in_flight = active.swap(true, Ordering::AcqRel);
        debug_assert!(
            !already_in_flight,
            "DesignOperator::apply_adjoint entered concurrently on one operator; \
             its shared scatter buffer is sound for only one in-flight call"
        );
        Self(active)
    }
}

#[cfg(debug_assertions)]
impl Drop for ReentryGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
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
        debug_assert_eq!(x.len(), self.design.n_dofs);
        debug_assert_eq!(y.len(), self.design.n_obs);
        gather_apply(self.design, x, y, self.sqrt_weights.as_deref());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        #[cfg(debug_assertions)]
        let _guard = ReentryGuard::acquire(&self.adjoint_active);
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

// The guard's flag is a private, debug-gated field, so this test lives beside
// it rather than in the crate's shared `operator/tests.rs`.
#[cfg(all(test, debug_assertions))]
mod reentry_guard_tests {
    use std::sync::atomic::Ordering;

    use schwarz_precond::Operator;

    use super::DesignOperator;
    use crate::domain::Design;
    use crate::observation::ObservationFrame;

    fn one_factor_design() -> Design<'static> {
        let frame = ObservationFrame::new(
            vec![vec![0u32, 1, 0]].into_iter().map(Into::into).collect(),
            Vec::new(),
        )
        .expect("valid frame");
        Design::from_frame(frame).expect("valid design")
    }

    #[test]
    #[should_panic(expected = "concurrently")]
    fn apply_adjoint_detects_in_flight_reentry() {
        let design = one_factor_design();
        let op = DesignOperator::new(&design, None);
        // Simulate a sibling apply_adjoint already in flight on this operator.
        op.adjoint_active.store(true, Ordering::Release);
        op.apply_adjoint(
            &vec![0.0; op.design.n_obs],
            &mut vec![0.0; op.design.n_dofs],
        )
        .expect("unreachable: the guard panics before returning");
    }
}
