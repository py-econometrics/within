//! Linear-algebra layer: the design operator (rectangular `sqrt(W) D`) plus the
//! Schwarz preconditioner construction in [`schwarz`].

pub(crate) mod design;
pub(crate) mod schwarz;
