//! A design's products over its term blocks: `y = s ⊙ Σ_t D_t x_t` and `x_t = D_tᵀ v`.

mod gather;
mod scatter;

pub(crate) use gather::gather;
pub(crate) use scatter::scatter;
