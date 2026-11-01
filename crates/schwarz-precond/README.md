# schwarz-precond

Generic domain decomposition library providing a one-level additive Schwarz
preconditioner with pluggable local solvers, plus a Modified LSMR iterative
solver for rectangular least-squares problems. Suitable for symmetric
positive (semi-)definite systems arising in finite elements, graph
Laplacians, and fixed-effects models.

## Install

```toml
[dependencies]
schwarz-precond = "0.1"
```

Or with Cargo:

```console
cargo add schwarz-precond
```

## Example

```rust
use faer::sparse::{SparseColMat, Triplet};
use schwarz_precond::{
    mlsmr, InitialGuess, LocalSolver, LsmrOptions, MatMut, MemBuffer, MemStack, Par,
    SchwarzPreconditioner, StackReq, SubdomainCore, SubdomainEntry,
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

// --- Diagonal local solver: x ← x / diag ---
struct DiagSolver(usize, f64);

impl LocalSolver for DiagSolver {
    fn n_local(&self) -> usize { self.0 }
    fn solve_local_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn solve_local(&self, mut x: MatMut<'_, f64>, _par: Par, _stack: &mut MemStack) {
        for j in 0..x.ncols() {
            for i in 0..x.nrows() { x[(i, j)] /= self.1; }
        }
    }
}

fn main() {
    let n = 100;
    let a = tridiag(n);
    let b = vec![1.0; n];

    // Build non-overlapping 2-DOF subdomains with uniform PoU weights
    let entries: Vec<_> = (0..n)
        .step_by(2)
        .map(|i| {
            let idx: Vec<u32> = (i..n.min(i + 2)).map(|j| j as u32).collect();
            let sz = idx.len();
            SubdomainEntry::try_new(SubdomainCore::uniform(idx), DiagSolver(sz, 3.0))
                .expect("valid subdomain entry")
        })
        .collect();

    let precond = SchwarzPreconditioner::new(entries);
    let options = LsmrOptions { tol: 1e-10, maxiter: 500, local_size: None, ..Default::default() };
    // The caller owns all per-run scratch; one buffer serves every solve it is passed to.
    let mut scratch = MemBuffer::new(StackReq::EMPTY);
    let mut x = vec![0.0; n];
    let info = mlsmr(&a, &b, &precond, &mut x, InitialGuess::Zero, &options, None, Par::rayon(0), &mut scratch)
    .expect("lsmr should converge");

    println!("converged={} iters={} res={:.3e}",
        info.converged, info.iterations, info.residual_norm);
}
```

For a full runnable example, see [`examples/additive_schwarz.rs`](examples/additive_schwarz.rs).

## License

MIT

## References

- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Smith, Bjørstad & Gropp (1996). *Domain Decomposition: Parallel Multilevel Methods for Elliptic PDEs*. Cambridge University Press.
- Nicolaides (1987). *Deflation of Conjugate Gradients with Applications to Boundary Value Problems*. SIAM J. Numer. Anal. 24(2).
