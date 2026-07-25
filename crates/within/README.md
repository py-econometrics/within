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
use within::{solve, Effect, LsmrOptions, PreconditionerConfig};

// Two factors: 100 levels each, 10 000 observations
let n_obs = 10_000usize;
let f0: Vec<u32> = (0..n_obs).map(|i| (i % 100) as u32).collect();
let f1: Vec<u32> = (0..n_obs).map(|i| (i / 100) as u32).collect();
let y: Vec<f64> = (0..n_obs).map(|i| i as f64 * 0.01).collect();
let design = vec![
    Effect::new(&f0, true, []).expect("valid effect"),
    Effect::new(&f1, true, []).expect("valid effect"),
];

// Solve with library defaults: LSMR + additive Schwarz
let result = solve(design.clone(), &y, None, &LsmrOptions::default(), None)
    .expect("solve should succeed");
assert!(result.converged);
println!("LSMR converged in {} iterations", result.iterations);

// Tighter tolerance with an explicit preconditioner config
let lsmr = LsmrOptions { tol: 1e-10, ..LsmrOptions::default() };
let precond = PreconditionerConfig::default();
let result = solve(design.clone(), &y, None, &lsmr, &precond)
    .expect("solve should succeed");
assert!(result.converged);

// Or opt into a diagonal/Jacobi preconditioner.
let diagonal = PreconditionerConfig::Diagonal;
let result = solve(design, &y, None, &lsmr, &diagonal)
    .expect("solve should succeed");
assert!(result.converged);
```

## Feature flags

| Feature   | Default | Effect                                                                 |
|-----------|---------|------------------------------------------------------------------------|
| `ndarray` | no      | Accepts `ndarray::ArrayView2<u32>` as a design, in addition to the always-available `Effect` slice terms. Off by default so the crate imposes no `ndarray` major version on callers. |

## Architecture

The crate is organized in four layers:

1. **`observation`** — Row-aligned observation columns via the
   `ObservationFrame` (categorical level codes plus continuous loadings,
   each borrowed or owned per column). Observation weights are not owned
   here — they flow as `Option<&[f64]>` to the operator layer.

2. **`domain`** — Domain decomposition. `Design` wraps a frame with
   factor metadata; `build_local_domains` constructs factor-pair subdomains
   with partition-of-unity weights for the Schwarz preconditioner.

3. **`operator`** — Linear algebra primitives. Internal rectangular
   `sqrt(W) D` operator for LSMR and Schwarz preconditioner builders
   that wire approximate Cholesky local solvers into the generic
   `schwarz-precond` framework.

4. **`orchestrate`** — End-to-end solve entry points (`solve`, `solve_batch`)
   with typed configuration (`LsmrOptions`, `PreconditionerConfig`).

## License

MIT

## References

- Correia, Sergio. "A feasible estimator for linear models with multi-way fixed effects." *Preprint* at http://scorreia.com/research/hdfe.pdf (2016).
- Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization. *SIAM Journal on Scientific Computing*.
- Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer.
- Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace Correction. *SIAM Review*, 34(4), 581--613.
