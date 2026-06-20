//! Custom local solver using faer dense Cholesky on local submatrices.
//!
//! Shows how to implement the `LocalSolver` trait with a dense factorization
//! of the local submatrix extracted from the global operator.

use faer::{MatRef, Side};

use schwarz_precond::{
    mlsmr, LocalSolveError, LocalSolver, Operator, ReductionStrategy, SchwarzPreconditioner,
    SolveError, SubdomainCore, SubdomainEntry,
};
// ---------------------------------------------------------------------------
// Tridiagonal operator
// ---------------------------------------------------------------------------

struct TridiagOperator {
    n: usize,
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
            y[i] = 3.0 * x[i];
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
// Dense Cholesky local solver (via faer)
// ---------------------------------------------------------------------------

struct DenseCholeskyLocalSolver {
    /// Row-major dense L factor from Cholesky
    l_row_major: Vec<f64>,
    n_local: usize,
}

impl DenseCholeskyLocalSolver {
    /// Build from a dense, row-major `m` x `m` local submatrix.
    fn from_dense(dense: &[f64], m: usize) -> Self {
        // Factor with faer: Cholesky L L^T
        let mat_ref = MatRef::from_row_major_slice(dense, m, m);
        let llt = mat_ref
            .llt(Side::Lower)
            .expect("Cholesky factorization failed (matrix not SPD)");

        // Store L back as row-major
        let l_mat = llt.L();
        let mut l_row_major = vec![0.0; m * m];
        for r in 0..m {
            for c in 0..m {
                l_row_major[r * m + c] = l_mat[(r, c)];
            }
        }

        Self {
            l_row_major,
            n_local: m,
        }
    }

    /// Solve L L^T x = rhs via forward/backward substitution.
    fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        let m = self.n_local;

        // Forward solve: L z = rhs
        let mut z = vec![0.0; m];
        for i in 0..m {
            let mut s = rhs[i];
            for (j, &zj) in z.iter().take(i).enumerate() {
                s -= self.l_row_major[i * m + j] * zj;
            }
            z[i] = s / self.l_row_major[i * m + i];
        }

        // Backward solve: L^T x = z
        for i in (0..m).rev() {
            let mut s = z[i];
            for (j, &sol_j) in sol.iter().enumerate().skip(i + 1) {
                s -= self.l_row_major[j * m + i] * sol_j;
            }
            sol[i] = s / self.l_row_major[i * m + i];
        }
    }
}

impl LocalSolver for DenseCholeskyLocalSolver {
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
        self.solve(rhs, sol);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Build subdomains with dense Cholesky solvers from local submatrices
// ---------------------------------------------------------------------------

/// Dense, row-major submatrix of the tridiagonal operator over `indices`.
///
/// The operator has `3` on the diagonal and `-1` between adjacent DOFs, so the
/// local block follows directly from the global index pattern — no sparse
/// container needed to carve it out.
fn tridiag_block(indices: &[usize]) -> Vec<f64> {
    let m = indices.len();
    let mut dense = vec![0.0; m * m];
    for (r, &global_r) in indices.iter().enumerate() {
        for (c, &global_c) in indices.iter().enumerate() {
            dense[r * m + c] = if global_r == global_c {
                3.0
            } else if global_r.abs_diff(global_c) == 1 {
                -1.0
            } else {
                0.0
            };
        }
    }
    dense
}

fn build_entries(n: usize) -> Vec<SubdomainEntry<DenseCholeskyLocalSolver>> {
    let mut entries = Vec::new();
    let mut i = 0;
    while i + 1 < n {
        let indices = vec![i, i + 1];
        let solver = DenseCholeskyLocalSolver::from_dense(&tridiag_block(&indices), indices.len());
        let global_indices = indices.iter().map(|&x| x as u32).collect();
        entries.push(
            SubdomainEntry::try_new(SubdomainCore::uniform(global_indices), solver)
                .expect("valid 2-DOF subdomain entry"),
        );
        i += 2;
    }
    if i < n {
        let indices = vec![i];
        let solver = DenseCholeskyLocalSolver::from_dense(&tridiag_block(&indices), indices.len());
        entries.push(
            SubdomainEntry::try_new(SubdomainCore::uniform(vec![i as u32]), solver)
                .expect("valid 1-DOF subdomain entry"),
        );
    }
    entries
}

fn main() {
    let n = 30;
    let rhs = vec![1.0; n];
    let a = TridiagOperator { n };

    let precond = SchwarzPreconditioner::new(build_entries(n), ReductionStrategy::default());
    let result = mlsmr(&a, &rhs, &precond, 1e-10, 200, None).expect("preconditioned lsmr");
    println!(
        "Dense Cholesky Schwarz LSMR: converged={}, iterations={:>3}, residual={:.3e}",
        result.converged, result.iterations, result.residual_norm,
    );
}
