# schwarz-precond

Generic domain decomposition library for additive and multiplicative Schwarz
preconditioners. Provides one-level variants with pluggable local solvers and
iterative solvers (CG, GMRES, LSMR). Suitable for symmetric positive
(semi-)definite systems arising in finite elements, graph Laplacians, and
fixed-effects models.

## Install

```toml
[dependencies]
schwarz-precond = "0.1"
```

Or with Cargo:

```
cargo add schwarz-precond
```

## Example

```rust
use schwarz_precond::{
    LocalSolver, Operator, SubdomainCore, SubdomainEntry,
    SchwarzPreconditioner,
};
use schwarz_precond::solve::cg::cg_solve_preconditioned;

// --- Tridiagonal SPD operator: A = tridiag(-1, 3, -1) ---
struct TridiagOp(usize);

impl Operator for TridiagOp {
    fn nrows(&self) -> usize { self.0 }
    fn ncols(&self) -> usize { self.0 }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.0 {
            y[i] = 3.0 * x[i];
            if i > 0          { y[i] -= x[i - 1]; }
            if i + 1 < self.0 { y[i] -= x[i + 1]; }
        }
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) { self.apply(x, y); }
}

// --- Diagonal local solver: sol = rhs / diag ---
struct DiagSolver(usize, f64);

impl LocalSolver for DiagSolver {
    fn n_local(&self) -> usize    { self.0 }
    fn scratch_size(&self) -> usize { self.0 }
    fn solve_local(&self, rhs: &mut [f64], sol: &mut [f64]) -> Result<(), schwarz_precond::LocalSolveError> {
        for i in 0..self.0 { sol[i] = rhs[i] / self.1; }
        Ok(())
    }
}

fn main() {
    let n = 100;
    let a = TridiagOp(n);
    let b = vec![1.0; n];

    // Build non-overlapping 2-DOF subdomains with uniform PoU weights
    let entries: Vec<_> = (0..n)
        .step_by(2)
        .map(|i| {
            let idx: Vec<u32> = (i..n.min(i + 2)).map(|j| j as u32).collect();
            let sz = idx.len();
            SubdomainEntry::new(SubdomainCore::uniform(idx), DiagSolver(sz, 3.0))
        })
        .collect();

    let precond = SchwarzPreconditioner::new(entries, n).expect("valid preconditioner");
    let result = cg_solve_preconditioned(&a, &precond, &b, 1e-10, 500).expect("cg should converge");

    println!("converged={} iters={} res={:.3e}",
        result.converged, result.iterations, result.residual_norm);
}
```

For a full runnable example, see [`examples/additive_schwarz.rs`](examples/additive_schwarz.rs).

## License

MIT

## References

- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Smith, Bjørstad & Gropp (1996). *Domain Decomposition: Parallel Multilevel Methods for Elliptic PDEs*. Cambridge University Press.
- Nicolaides (1987). *Deflation of Conjugate Gradients with Applications to Boundary Value Problems*. SIAM J. Numer. Anal. 24(2).
