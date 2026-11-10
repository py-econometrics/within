//! Linear-algebra layer: the design operator (rectangular `sqrt(W) D`) plus the
//! Schwarz preconditioner construction in [`schwarz`].

pub(crate) mod coarse;
pub(crate) mod design;
#[cfg(feature = "pair-schur-prototype")]
pub(crate) mod reduced_coarse;
#[cfg(feature = "pair-schur-prototype")]
pub(crate) mod reduced_design;
pub(crate) mod schwarz;
