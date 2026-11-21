//! `D̃`'s products over its term blocks, `y = s ⊙ Σ_t D̃_t x_t` and `x_t = D̃_tᵀ v`: kernels that
//! read the design.

mod gather;
mod scatter;

pub(super) use gather::gather;
pub(super) use scatter::{fold_workspaces, scatter};
