# within

Solves linear fixed-effects models **y = D x + e** where D is a sparse
categorical design matrix. Modified LSMR with a domain-decomposition
(Schwarz) preconditioner backed by approximate Cholesky local solvers.
Designed for econometric panel data with millions of observations and
multiple high-dimensional fixed effects.

## Install

```
cargo add within
```

## Quick example

```rust
use ndarray::Array2;
use within::{solve, LsmrOptions, PreconditionerConfig};

// Two factors: 100 levels each, 10 000 observations
let n_obs = 10_000usize;
let mut categories = Array2::<u32>::zeros((n_obs, 2));
for i in 0..n_obs {
    categories[[i, 0]] = (i % 100) as u32;
    categories[[i, 1]] = (i / 100) as u32;
}
let y: Vec<f64> = (0..n_obs).map(|i| i as f64 * 0.01).collect();

// Solve with library defaults: LSMR + the adaptive diagonal→Schwarz ladder
let result = solve(categories.view(), &y, None, &LsmrOptions::default(), None)
    .expect("solve should succeed");
assert!(result.converged);
println!("LSMR converged in {} iterations", result.iterations);

// Tighter tolerance, library-default preconditioner
let lsmr = LsmrOptions { tol: 1e-10, ..LsmrOptions::default() };
let precond = PreconditionerConfig::default();
let result = solve(categories.view(), &y, None, &lsmr, &precond)
    .expect("solve should succeed");
assert!(result.converged);

// Or opt into a diagonal/Jacobi preconditioner.
let diagonal = PreconditionerConfig::Diagonal;
let result = solve(categories.view(), &y, None, &lsmr, &diagonal)
    .expect("solve should succeed");
assert!(result.converged);
```

## Architecture

The crate is organized in four layers, each depending only on the ones above it:

1. **`design`** — The weight-free design `D = E • X`. `Design` lowers
   `Effect`s (`Design::new`) or an observation-major category matrix
   (`Design::from_categories`) into the shared incidence `E` (per-term level
   codes and coefficient layout) and each term's slope columns `X`, and
   provides the products `D x` and `Dᵀ v` over its term blocks.

2. **`system`** — The least-squares system for one weighting `W`.
   `OrthogonalDesign` is `D̃ = D·R`, each term's slopes orthogonalized within
   each level; `DesignOperator` is `A = sqrt(W) D̃`, the operator LSMR
   iterates on; `Gramian` is `G = AᵀA`, which the preconditioners read.

3. **`precond`** — `M ≈ G⁻¹`: the diagonal and the additive Schwarz
   preconditioner, whose factor-pair subdomains (`build_local_domains`) wire
   approximate Cholesky local solvers into the generic `schwarz-precond`
   framework.

4. **`solver`** — `Solver` (a design and its preconditioner, reused across
   right-hand sides) and the one-shot entry points `solve` and `solve_batch`,
   with typed configuration (`LsmrOptions`, `PreconditionerConfig`).

## License

MIT

## References

- Correia, Sergio. "A feasible estimator for linear models with multi-way fixed effects." *Preprint* at http://scorreia.com/research/hdfe.pdf (2016).
- Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization. *SIAM Journal on Scientific Computing*.
- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace Correction. *SIAM Review*, 34(4), 581--613.
