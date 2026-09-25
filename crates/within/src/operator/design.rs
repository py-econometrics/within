use std::borrow::Cow;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};

use portable_atomic::AtomicF64;
use schwarz_precond::Operator;

use crate::domain::PreparedDesign;

mod gather;
mod scatter;

#[cfg(test)]
mod tests;

use gather::{gather_apply, unscale};
use scatter::scatter_apply;

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Design operator `D` or `W^{1/2} D`, whose normal equations `AᵀA = DᵀWD` recover the Gramian.
pub(crate) struct DesignOperator<'a> {
    prepared: &'a PreparedDesign<'a>,
    /// Sized once to the largest term's block, so it allocates per operator, not per iteration.
    scatter_scratch: Vec<AtomicF64>,
    /// Debug-only reentry sentinel: a concurrent `apply_adjoint` would race the scratch writes.
    #[cfg(debug_assertions)]
    adjoint_active: AtomicBool,
}

impl<'a> DesignOperator<'a> {
    pub(crate) fn new(prepared: &'a PreparedDesign<'a>) -> Self {
        let design = &prepared.design;
        let max_block = design.terms.iter().map(|t| t.n_dofs()).max().unwrap_or(0);
        Self {
            prepared,
            scatter_scratch: (0..max_block).map(|_| AtomicF64::new(0.0)).collect(),
            #[cfg(debug_assertions)]
            adjoint_active: AtomicBool::new(false),
        }
    }

    /// Squared column norms `diag(AᵀA)`.
    pub(crate) fn column_norms_squared(&self) -> Vec<f64> {
        let design = &self.prepared.design;
        let mut diag = vec![0.0; design.n_dofs];
        for (index, term) in design.terms.iter().enumerate() {
            let levels = design.frame.level_column(index);
            for (column, loading) in term.columns.iter().enumerate() {
                let z = loading
                    .covariate()
                    .map(|&c| self.prepared.loading_column(c as usize));
                for (obs, &level) in levels.iter().enumerate() {
                    let w = self.prepared.row_weight(obs);
                    // Keep `w * z * z` left-to-right: a zero weight kills a huge `z` first.
                    diag[term.dof_index(column, level as usize)] +=
                        z.map_or(w, |z| w * z[obs] * z[obs]);
                }
            }
        }
        diag
    }

    /// `y − D x`, read off this operator's measured `b − A x` unless a zero weight erased a row.
    pub(crate) fn demeaned(&self, x: &[f64], y: &[f64], measured: Option<Vec<f64>>) -> Vec<f64> {
        match (measured, self.prepared.sqrt_weights()) {
            (Some(residual), None) => residual,
            (Some(mut residual), Some(sw)) if !sw.contains(&0.0) => {
                unscale(&mut residual, sw);
                residual
            }
            (measured, _) => {
                let mut demeaned = measured.unwrap_or_else(|| vec![0.0; y.len()]);
                gather_apply(self.prepared, x, &mut demeaned, None);
                for (d, &yi) in demeaned.iter_mut().zip(y) {
                    *d = yi - *d;
                }
                demeaned
            }
        }
    }

    /// Observation-space RHS `b = W^{1/2} y`; borrows unweighted, owns weighted.
    pub(crate) fn weighted_rhs<'y>(&self, y: &'y [f64]) -> Cow<'y, [f64]> {
        match self.prepared.sqrt_weights() {
            None => Cow::Borrowed(y),
            Some(sw) => Cow::Owned(y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect()),
        }
    }
}

/// RAII reentry guard; `Drop` clears the flag on every exit path including panics.
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
        self.prepared.design.n_obs
    }

    fn ncols(&self) -> usize {
        self.prepared.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.prepared.design.n_dofs);
        debug_assert_eq!(y.len(), self.prepared.design.n_obs);
        gather_apply(self.prepared, x, y, self.prepared.sqrt_weights());
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        #[cfg(debug_assertions)]
        let _guard = ReentryGuard::acquire(&self.adjoint_active);
        debug_assert_eq!(x.len(), self.prepared.design.n_obs);
        debug_assert_eq!(y.len(), self.prepared.design.n_dofs);
        y.fill(0.0);
        // No lock needed: `solve_batch` builds one operator per RHS, so calls are sequential.
        match self.prepared.sqrt_weights() {
            Some(sw) => scatter_apply(self.prepared, &self.scatter_scratch, y, &|i| sw[i] * x[i]),
            None => scatter_apply(self.prepared, &self.scatter_scratch, y, &|i| x[i]),
        }
        Ok(())
    }
}

// The guard's flag is private and debug-gated, so this test lives beside it.
#[cfg(all(test, debug_assertions))]
mod reentry_guard_tests {
    use std::sync::atomic::Ordering;

    use schwarz_precond::Operator;

    use super::DesignOperator;
    use crate::domain::PreparedDesign;

    #[test]
    #[should_panic(expected = "concurrently")]
    fn apply_adjoint_detects_in_flight_reentry() {
        let design = PreparedDesign::from_levels_for_test(vec![vec![0, 1, 0]]);
        let op = DesignOperator::new(&design);
        // Simulate a sibling `apply_adjoint` already in flight on this operator.
        op.adjoint_active.store(true, Ordering::Release);
        op.apply_adjoint(
            &vec![0.0; op.prepared.design.n_obs],
            &mut vec![0.0; op.prepared.design.n_dofs],
        )
        .expect("unreachable: the guard panics before returning");
    }
}
