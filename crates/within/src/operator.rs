//! Linear-algebra layer: the design operator (rectangular `sqrt(W) D`) plus the
//! Schwarz preconditioner construction in [`schwarz`].

pub(crate) mod design;
pub(crate) mod schwarz;
pub(crate) mod two_level;

/// `y = A x` on the whole pool, its scratch allocated.
#[cfg(test)]
pub(crate) fn apply_for_test<O: schwarz_precond::Operator + ?Sized>(
    op: &O,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), schwarz_precond::SolveError> {
    use schwarz_precond::{MemBuffer, MemStack, Threads};
    let mut scratch = MemBuffer::new(op.scratch(1, Threads::All));
    op.apply(x, y, Threads::All, MemStack::new(&mut scratch))
}

/// `y = Aᵀ x` on the whole pool, its scratch allocated.
#[cfg(test)]
pub(crate) fn apply_adjoint_for_test<O: schwarz_precond::Operator + ?Sized>(
    op: &O,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), schwarz_precond::SolveError> {
    use schwarz_precond::{MemBuffer, MemStack, Threads};
    let mut scratch = MemBuffer::new(op.scratch(1, Threads::All));
    op.apply_adjoint(x, y, Threads::All, MemStack::new(&mut scratch))
}
