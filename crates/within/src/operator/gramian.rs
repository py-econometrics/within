//! Gramian `G = D^T W D` construction, operations, and operator wrappers.

mod accumulator;
mod cross_tab;
mod csr_assembly;
mod explicit;
mod implicit;

pub(crate) use cross_tab::CrossTab;
pub use implicit::GramianOperator;

use std::sync::Arc;

use schwarz_precond::SparseMatrix;

/// Explicit Gramian `G = D^T W D` stored as CSR.
pub struct Gramian {
    pub matrix: Arc<SparseMatrix>,
}
