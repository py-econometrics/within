//! Shared test fixtures for schwarz-precond integration tests.

use schwarz_precond::{
    LocalSolveError, LocalSolver, Operator, SolveError, SubdomainCore, SubdomainEntry,
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
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for i in 0..self.n {
            y[i] = self.diag_val * x[i];
            if i > 0 {
                y[i] -= x[i - 1];
            }
            if i + 1 < self.n {
                y[i] -= x[i + 1];
            }
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
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
        _allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        for i in 0..self.n_local {
            sol[i] = rhs[i] * self.inv_diag;
        }
        Ok(())
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

/// Local solver that always fails with `BackendFailed`.
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
        _allow_inner_parallelism: bool,
    ) -> Result<(), LocalSolveError> {
        Err(LocalSolveError::BackendFailed {
            context: "test.failing_local_solver",
            message: format!("deliberate failure for n={}", self.n_local),
        })
    }
}
