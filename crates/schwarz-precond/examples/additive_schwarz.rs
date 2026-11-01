//! Additive Schwarz preconditioned MLSMR on a tridiagonal system.
//!
//! Demonstrates the simplest use of `SchwarzPreconditioner` with hand-built
//! subdomains and diagonal local solvers.

use faer::sparse::{SparseColMat, Triplet};
use schwarz_precond::{
    mlsmr, IdentityPrecond, InitialGuess, LocalSolver, LsmrOptions, MatMut, MemBuffer, MemStack,
    Par, SchwarzPreconditioner, StackReq, SubdomainCore, SubdomainEntry,
};

/// `tridiag(-1, 3, -1)` of size `n`, as a faer sparse matrix (a `BiLinOp`).
fn tridiag(n: usize) -> SparseColMat<usize, f64> {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push(Triplet::new(i, i, 3.0));
        if i + 1 < n {
            triplets.push(Triplet::new(i, i + 1, -1.0));
            triplets.push(Triplet::new(i + 1, i, -1.0));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).expect("valid triplets")
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
    fn solve_local_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn solve_local(&self, mut x: MatMut<'_, f64>, _par: Par, _stack: &mut MemStack) {
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                x[(i, j)] /= self.diag_val;
            }
        }
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
    let a = tridiag(n);
    let options = LsmrOptions {
        tol: 1e-10,
        maxiter: 200,
        local_size: None,
        ..Default::default()
    };
    // One scratch buffer, grown on first use, serves every solve.
    let mut scratch = MemBuffer::new(StackReq::EMPTY);

    // --- Unpreconditioned LSMR ---
    let mut x = vec![0.0; n];
    let result_plain = mlsmr(
        &a,
        &rhs,
        &IdentityPrecond { dim: x.len() },
        &mut x,
        InitialGuess::Zero,
        &options,
        None,
        Par::Seq,
        &mut scratch,
    )
    .expect("unpreconditioned lsmr");
    println!(
        "Unpreconditioned LSMR : converged={}, iterations={:>3}, residual={:.3e}",
        result_plain.converged, result_plain.iterations, result_plain.residual_norm,
    );

    // --- Additive Schwarz preconditioned LSMR ---
    let precond = SchwarzPreconditioner::new(build_entries(n));
    x.fill(0.0);
    let result_schwarz = mlsmr(
        &a,
        &rhs,
        &precond,
        &mut x,
        InitialGuess::Zero,
        &options,
        None,
        Par::rayon(0),
        &mut scratch,
    )
    .expect("preconditioned lsmr");
    println!(
        "Additive Schwarz LSMR : converged={}, iterations={:>3}, residual={:.3e}",
        result_schwarz.converged, result_schwarz.iterations, result_schwarz.residual_norm,
    );
}
