//! Schwarz preconditioners: additive and multiplicative variants.
//!
//! Both variants implement the [`Operator`](crate::Operator) trait, so they
//! can be passed directly to CG or GMRES as a preconditioner.
//!
//! - [`additive`] — `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`: independent local solves
//!   combined via atomic scatter or parallel reduction. Symmetric, so valid
//!   for both CG and GMRES.
//! - [`multiplicative`] — applies subdomains sequentially, each seeing
//!   the updated residual from all preceding solves. For a forward sweep
//!   over N subdomains the operator is the product:
//!
//!   ```text
//!   M⁻¹ = (I - Tₙ)(I - Tₙ₋₁)···(I - T₁)    where Tᵢ = Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ A
//!   ```
//!
//!   This is the block Gauss-Seidel analogue: it converges faster per
//!   iteration than additive but is non-symmetric, requiring GMRES.

mod additive;
mod multiplicative;

pub use additive::{AdditiveSchwarzDiagnostics, ReductionStrategy, SchwarzPreconditioner};
pub use multiplicative::{
    MultiplicativeSchwarzPreconditioner, OperatorResidualUpdater, ResidualUpdater,
};
