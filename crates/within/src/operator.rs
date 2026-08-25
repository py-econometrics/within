//! Linear-algebra layer: [`DesignOperator`] (rectangular `sqrt(W) D`) plus the
//! Schwarz preconditioner construction in [`schwarz`].

pub(crate) mod design;
pub(crate) mod schwarz;

pub(crate) use design::DesignOperator;
