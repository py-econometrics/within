//! Additive Schwarz preconditioner `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ` as an
//! [`LinOp`](faer::matrix_free::LinOp).

mod preconditioner;

pub use preconditioner::SchwarzPreconditioner;
