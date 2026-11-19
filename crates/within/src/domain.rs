//! Domain layer: the design `D = E • X`, its orthogonalization `D̃` for one weighting, and `G`.

mod design;
mod incidence;
mod orthogonal;

pub use design::{Design, Effect};
pub use incidence::CoefficientLayout;
pub(crate) use incidence::{
    Column, FactorEncoding, Incidence, LevelSlots, RowIndex, ScatterSplit, TermLayout, TermRows,
};
pub(crate) use orthogonal::{Gramian, OrthogonalDesign, RowKernel, TermBlock};

/// Rows from which a sweep over observations runs in parallel.
pub(crate) const PAR_THRESHOLD: usize = 10_000;
