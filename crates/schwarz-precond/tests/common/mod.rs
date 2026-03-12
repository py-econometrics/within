//! Shared test fixtures for schwarz-precond integration tests.

#![allow(dead_code)]

use schwarz_precond::{
    LocalSolveError, LocalSolveOptions, LocalSolver, Operator, SparseMatrix, SubdomainCore,
    SubdomainEntry,
};

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Tridiagonal SPD operator: diag=`diag_val`, off-diag=-1.
pub struct TridiagOperator {
    pub n: usize,
    pub diag_val: f64,
}

impl TridiagOperator {
    pub fn new(n: usize, diag_val: f64) -> Self {
        Self { n, diag_val }
    }
}

impl Operator for TridiagOperator {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            y[i] = self.diag_val * x[i];
            if i > 0 {
                y[i] -= x[i - 1];
            }
            if i + 1 < self.n {
                y[i] -= x[i + 1];
            }
        }
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.apply(x, y);
    }
}

/// Simple diagonal operator.
pub struct DiagOperator {
    pub values: Vec<f64>,
}

impl Operator for DiagOperator {
    fn nrows(&self) -> usize {
        self.values.len()
    }
    fn ncols(&self) -> usize {
        self.values.len()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.values.len() {
            y[i] = self.values[i] * x[i];
        }
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.apply(x, y);
    }
}

/// Identity operator.
pub struct IdentityOp {
    pub n: usize,
}

impl Operator for IdentityOp {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        y.copy_from_slice(x);
    }
}

/// 3x3 SPD tridiagonal: [[4,1,0],[1,3,1],[0,1,2]]
pub struct SpdMatrix3;

impl Operator for SpdMatrix3 {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        3
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        y[0] = 4.0 * x[0] + 1.0 * x[1];
        y[1] = 1.0 * x[0] + 3.0 * x[1] + 1.0 * x[2];
        y[2] = 1.0 * x[1] + 2.0 * x[2];
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.apply(x, y);
    }
}

/// Jacobi (diagonal) preconditioner for SpdMatrix3.
pub struct JacobiPrecond3;

impl Operator for JacobiPrecond3 {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        3
    }
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        z[0] = r[0] / 4.0;
        z[1] = r[1] / 3.0;
        z[2] = r[2] / 2.0;
    }
    fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) {
        self.apply(r, z);
    }
}

/// Nonsymmetric 3x3 matrix: [[3,1,0],[0,4,2],[1,0,5]]
pub struct NonsymMatrix3;

impl Operator for NonsymMatrix3 {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        3
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        y[0] = 3.0 * x[0] + 1.0 * x[1];
        y[1] = 4.0 * x[1] + 2.0 * x[2];
        y[2] = 1.0 * x[0] + 5.0 * x[2];
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        y[0] = 3.0 * x[0] + 1.0 * x[2];
        y[1] = 1.0 * x[0] + 4.0 * x[1];
        y[2] = 2.0 * x[1] + 5.0 * x[2];
    }
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
    fn scratch_size(&self) -> usize {
        self.n_local
    }
    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        _options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        for i in 0..self.n_local {
            sol[i] = rhs[i] * self.inv_diag;
        }
        Ok(())
    }
}

/// Exact inverse local solver for per-element diagonal systems.
pub struct DiagLocalSolver {
    pub inv_diag: Vec<f64>,
}

impl DiagLocalSolver {
    pub fn new(diag: &[f64]) -> Self {
        Self {
            inv_diag: diag.iter().map(|&d| 1.0 / d).collect(),
        }
    }
}

impl LocalSolver for DiagLocalSolver {
    fn n_local(&self) -> usize {
        self.inv_diag.len()
    }
    fn scratch_size(&self) -> usize {
        self.inv_diag.len()
    }
    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        _options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        for i in 0..self.inv_diag.len() {
            sol[i] = self.inv_diag[i] * rhs[i];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

/// Build a path graph (0 -- 1 -- ... -- n-1) as a SparseMatrix.
pub fn path_graph(n: usize) -> SparseMatrix {
    let mut indptr = vec![0u32; n + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        let mut row_nnz = 0;
        if i > 0 {
            indices.push((i - 1) as u32);
            data.push(1.0);
            row_nnz += 1;
        }
        if i + 1 < n {
            indices.push((i + 1) as u32);
            data.push(1.0);
            row_nnz += 1;
        }
        indptr[i + 1] = indptr[i] + row_nnz;
    }
    SparseMatrix::new(indptr, indices, data, n)
}

/// Build a cycle graph (0 -- 1 -- ... -- n-1 -- 0) as a SparseMatrix.
pub fn cycle_graph(n: usize) -> SparseMatrix {
    let mut indptr = vec![0u32; n + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = (i + 1) % n;
        let (a, b) = if prev < next {
            (prev, next)
        } else {
            (next, prev)
        };
        indices.push(a as u32);
        data.push(1.0);
        indices.push(b as u32);
        data.push(1.0);
        indptr[i + 1] = indptr[i] + 2;
    }
    SparseMatrix::new(indptr, indices, data, n)
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

/// Local solver that always fails with `ApproxCholSolveFailed`.
pub struct FailingLocalSolver {
    pub n_local: usize,
    pub scratch_size: usize,
}

impl LocalSolver for FailingLocalSolver {
    fn n_local(&self) -> usize {
        self.n_local
    }
    fn scratch_size(&self) -> usize {
        self.scratch_size
    }
    fn solve_local(
        &self,
        _rhs: &mut [f64],
        _sol: &mut [f64],
        _options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        Err(LocalSolveError::ApproxCholSolveFailed {
            context: "test.failing_local_solver",
            message: format!("deliberate failure for n={}", self.n_local),
        })
    }
}

/// Check that ||Ax - b|| < tol.
pub fn check_residual<A: Operator>(op: &A, x: &[f64], b: &[f64], tol: f64) {
    let n = b.len();
    let mut ax = vec![0.0; n];
    op.apply(x, &mut ax);
    let err: f64 = ax
        .iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(err < tol, "residual too large: {err}");
}
