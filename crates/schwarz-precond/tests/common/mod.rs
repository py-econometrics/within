//! Shared test fixtures for schwarz-precond integration tests.

use schwarz_precond::{
    BiLinOp, LinOp, LocalSolver, MatMut, MatRef, MemBuffer, MemStack, Par, StackReq, SubdomainCore,
    SubdomainEntry,
};

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Tridiagonal SPD operator: diag=`diag_val`, off-diag=-1.
#[derive(Debug)]
pub struct TridiagOperator {
    pub n: usize,
    pub diag_val: f64,
}

impl TridiagOperator {
    pub fn new(n: usize, diag_val: f64) -> Self {
        Self { n, diag_val }
    }
}

impl LinOp<f64> for TridiagOperator {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, _: Par, _: &mut MemStack) {
        for j in 0..rhs.ncols() {
            for i in 0..self.n {
                let mut y = self.diag_val * rhs[(i, j)];
                if i > 0 {
                    y -= rhs[(i - 1, j)];
                }
                if i + 1 < self.n {
                    y -= rhs[(i + 1, j)];
                }
                out[(i, j)] = y;
            }
        }
    }
    fn conj_apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, s: &mut MemStack) {
        LinOp::apply(self, out, rhs, par, s)
    }
}

/// Symmetric, so every adjoint is the apply.
impl BiLinOp<f64> for TridiagOperator {
    fn transpose_apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn transpose_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        s: &mut MemStack,
    ) {
        LinOp::apply(self, out, rhs, par, s)
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        s: &mut MemStack,
    ) {
        LinOp::apply(self, out, rhs, par, s)
    }
}

/// `z = op r` under `par`, on scratch sized by the operator.
pub fn apply(op: &impl LinOp<f64>, r: &[f64], z: &mut [f64], par: Par) {
    let mut scratch = MemBuffer::new(op.apply_scratch(1, par));
    let n = z.len();
    op.apply(
        MatMut::from_column_major_slice_mut(z, n, 1),
        MatRef::from_column_major_slice(r, r.len(), 1),
        par,
        MemStack::new(&mut scratch),
    );
}

// ---------------------------------------------------------------------------
// Local solvers
// ---------------------------------------------------------------------------

/// Diagonal local solver: y = rhs / diag_val (uniform diagonal).
pub struct UniformDiagLocalSolver {
    pub n_local: usize,
    pub inv_diag: f64,
}

impl UniformDiagLocalSolver {
    pub fn new(n_local: usize, diag_val: f64) -> Self {
        Self {
            n_local,
            inv_diag: 1.0 / diag_val,
        }
    }
}

impl LocalSolver for UniformDiagLocalSolver {
    fn n_local(&self) -> usize {
        self.n_local
    }
    fn solve_local_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn solve_local(&self, mut x: MatMut<'_, f64>, _par: Par, _stack: &mut MemStack) {
        assert_eq!(x.nrows(), self.n_local);
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                x[(i, j)] *= self.inv_diag;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build non-overlapping 2-DOF subdomain entries covering `n` DOFs.
pub fn make_schwarz_entries(n: usize) -> Vec<SubdomainEntry<UniformDiagLocalSolver>> {
    let mut entries = Vec::new();
    let mut i = 0;
    while i + 1 < n {
        entries.push(
            SubdomainEntry::try_new(
                SubdomainCore::uniform(vec![i as u32, (i + 1) as u32]),
                UniformDiagLocalSolver::new(2, 3.0),
            )
            .expect("valid 2-DOF subdomain entry"),
        );
        i += 2;
    }
    if i < n {
        entries.push(
            SubdomainEntry::try_new(
                SubdomainCore::uniform(vec![i as u32]),
                UniformDiagLocalSolver::new(1, 3.0),
            )
            .expect("valid 1-DOF subdomain entry"),
        );
    }
    entries
}
