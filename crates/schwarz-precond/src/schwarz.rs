//! Schwarz preconditioners: additive and multiplicative variants.

mod additive;
mod multiplicative;

pub use additive::{AdditiveSchwarzDiagnostics, ReductionStrategy, SchwarzPreconditioner};
pub use multiplicative::{
    MultiplicativeSchwarzPreconditioner, OperatorResidualUpdater, ResidualUpdater,
};
