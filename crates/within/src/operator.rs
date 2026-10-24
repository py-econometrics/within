//! Linear-algebra layer: [`DesignOperator`] (rectangular `sqrt(W) D`) plus the
//! Schwarz preconditioner construction in [`schwarz`].

pub(crate) mod design;
#[cfg(feature = "pair-schur-prototype")]
pub(crate) mod reduced_design;
pub(crate) mod schwarz;

pub(crate) use design::DesignOperator;
