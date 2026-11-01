//! Linear-algebra layer: the design operator (rectangular `sqrt(W) D`) plus the
//! Schwarz preconditioner construction in [`schwarz`].

pub(crate) mod design;
pub(crate) mod schwarz;

#[cfg(test)]
use schwarz_precond::BiLinOp;
use schwarz_precond::{ColMut, ColRef, LinOp, MemBuffer, MemStack, Par};

/// `y = op x` on the ambient pool with fresh scratch: the slice boundary of a [`LinOp`].
pub(crate) fn apply_ambient(op: &impl LinOp<f64>, x: &[f64], y: &mut [f64]) {
    let par = Par::rayon(0);
    let mut scratch = MemBuffer::new(op.apply_scratch(1, par));
    let (x, y) = (
        ColRef::from_slice(x).as_mat(),
        ColMut::from_slice_mut(y).as_mat_mut(),
    );
    op.apply(y, x, par, MemStack::new(&mut scratch));
}

/// `y = opᵀ x` on the ambient pool with fresh scratch.
#[cfg(test)]
pub(crate) fn transpose_apply_ambient(op: &impl BiLinOp<f64>, x: &[f64], y: &mut [f64]) {
    let par = Par::rayon(0);
    let mut scratch = MemBuffer::new(op.transpose_apply_scratch(1, par));
    let (x, y) = (
        ColRef::from_slice(x).as_mat(),
        ColMut::from_slice_mut(y).as_mat_mut(),
    );
    op.transpose_apply(y, x, par, MemStack::new(&mut scratch));
}
