//! A coarse correction composed around the Schwarz map: the adaptive ladder's last rung.

use schwarz_precond::{Operator, SolveError};

use super::schwarz::Preconditioner;

/// A correction applied around an inner map, built once that map stalls.
pub(crate) trait CoarseRung: Send + Sync {
    fn apply_around(
        &self,
        inner: &dyn Operator,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), SolveError>;

    /// Probe: `k` interleaved columns around a block `inner`.
    fn apply_around_block(
        &self,
        _inner: &(dyn Fn(&[f64], &mut [f64]) + Sync),
        _x: &[f64],
        _y: &mut [f64],
        _k: usize,
    ) {
        unimplemented!("probe: no block coarse apply")
    }
}

/// The Schwarz map with a coarse rung around it.
pub(crate) struct Coarsened<'a> {
    pub(crate) inner: &'a Preconditioner,
    pub(crate) coarse: &'a dyn CoarseRung,
}

impl Operator for Coarsened<'_> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.coarse.apply_around(self.inner, x, y)
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.coarse.apply_around(self.inner, x, y)
    }
}
