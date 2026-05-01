//! Additive Schwarz preconditioned CG on a tridiagonal system.
//!
//! Demonstrates the simplest use of `SchwarzPreconditioner` with hand-built
//! subdomains and diagonal local solvers.

use schwarz_precond::solve::cg::pcg;
use schwarz_precond::{
    LocalSolver, Operator, SchwarzPreconditioner, SolveError, SubdomainCore, SubdomainEntry,
};

// ---------------------------------------------------------------------------
// Tridiagonal operator: diag=3, off-diag=-1
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
        self.apply(x, y) // symmetric
    }
}

// ---------------------------------------------------------------------------
// Diagonal local solver: y = rhs / diag_val
// ---------------------------------------------------------------------------

struct DiagLocalSolver {
    n_local: usize,
    diag_val: f64,
}

impl LocalSolver for DiagLocalSolver {
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
    ) -> Result<(), SolveError> {
        for i in 0..self.n_local {
            sol[i] = rhs[i] / self.diag_val;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Build ~n/2 subdomains of size 2 covering [0..n)
// ---------------------------------------------------------------------------

fn build_entries(n: usize) -> Vec<SubdomainEntry<DiagLocalSolver>> {
    let mut entries = Vec::new();
    let mut i = 0;
    while i + 1 < n {
        entries.push(
            SubdomainEntry::try_new(
                SubdomainCore::uniform(vec![i as u32, (i + 1) as u32]),
                DiagLocalSolver {
                    n_local: 2,
                    diag_val: 3.0,
                },
            )
            .expect("valid 2-DOF subdomain entry"),
        );
        i += 2;
    }
    // Handle odd n: last DOF gets a size-1 subdomain
    if i < n {
        entries.push(
            SubdomainEntry::try_new(
                SubdomainCore::uniform(vec![i as u32]),
                DiagLocalSolver {
                    n_local: 1,
                    diag_val: 3.0,
                },
            )
            .expect("valid 1-DOF subdomain entry"),
        );
    }
    entries
}

fn main() {
    let n = 20;
    let rhs = vec![1.0; n];
    let a = TridiagOperator { n };

    // --- Unpreconditioned CG ---
    let result_plain =
        pcg(&a, &rhs, None::<&TridiagOperator>, 1e-10, 200).expect("unpreconditioned cg");
    println!(
        "Unpreconditioned CG : converged={}, iterations={:>3}, residual={:.3e}",
        result_plain.converged, result_plain.iterations, result_plain.residual_norm,
    );

    // --- Additive Schwarz preconditioned CG ---
    let precond = SchwarzPreconditioner::new(build_entries(n), n)
        .expect("valid additive schwarz preconditioner");
    let result_schwarz = pcg(&a, &rhs, Some(&precond), 1e-10, 200).expect("preconditioned cg");
    println!(
        "Additive Schwarz CG : converged={}, iterations={:>3}, residual={:.3e}",
        result_schwarz.converged, result_schwarz.iterations, result_schwarz.residual_norm,
    );
}
