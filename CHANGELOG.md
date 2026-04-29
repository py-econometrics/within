# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking Changes

> All items in this section break source compatibility for Rust consumers
> of the `within` and `schwarz-precond` crates. The Python boundary
> (`within.solve`, `within.Solver`, type stubs) is unchanged.

- **`schwarz_precond::SolveError` is now `#[non_exhaustive]`** and gained an
  `InvalidInput { context, message }` variant for pre-iteration validation
  failures. Downstream Rust consumers matching on `SolveError` must add a
  wildcard arm. Future variant additions will not be breaking.
- **Observation store API: `ObservationStore` → `Store`; weights are
  externalized.** The `Store` trait no longer carries weights — `weight()`
  and `is_unweighted()` are removed and the `ObservationWeights` enum is
  deleted. `FactorMajorStore::new` and `ArrayStore::new` no longer accept
  a weights argument. Weight-vs-design length validation now happens at
  `Solver` construction (`validate_weights`), not at store construction.
- **Domain API: `WeightedDesign` → `Design`; matvec/gather/scatter helpers
  removed.** `Design` is now pure data + factor metadata. The methods
  `matvec_d`, `rmatvec_dt`, `rmatvec_wdt`, `gather_add`, `scatter_add`,
  `uid_weight`, `is_unweighted`, and `gramian_diagonal` are gone. Migrate
  call sites to the operator types:
  - `design.matvec_d(x, y)` → `DesignOperator::new(&design).apply(x, y)`
  - `design.rmatvec_dt(r, x)` →
    `DesignOperator::new(&design).apply_adjoint(r, x)`
  - `design.gramian_diagonal()` → `Gramian::build(&design).diagonal()`
    (or `Gramian::build_weighted(&design, &w).diagonal()` when weighted).
- **Operator layer split into weighted / unweighted variants.** The
  implicit Gramian splits into `GramianOperator` (`D^T D`) and the new
  `WeightedGramianOperator` (`D^T W D`); the design operator splits into
  `DesignOperator` (`D`) and the new `WeightedDesignOperator` (`sqrt(W) D`,
  used by the LSMR path). Each Gramian operator exposes `to_csr()`.
  Explicit assembly is now `Gramian::build` (unweighted) and
  `Gramian::build_weighted` (weighted). Weighted apply / apply\_adjoint
  are single fused passes — the `Mutex` scratch buffer is gone.
- **`Solver::from_design` and `Solver::from_design_with_preconditioner`
  take an explicit `weights: Option<Vec<f64>>` argument** between
  `design` and `params`, mirroring the externalized weights. Passing
  `None` selects the unweighted operator pair at solve time.
- **Internal builders take `Option<&[f64]>` weights:** `build_schwarz`,
  `build_preconditioner`, `CrossTab::build_for_pair`, and the
  `domain::factor_pairs` constructors. Callers using these lower-level
  entry points directly must now pass weights explicitly.

### Added

- **LSMR rectangular least-squares solver**: a preconditioned LSMR variant
  operating directly on the weighted design operator (`sqrt(W) D`), exposed
  via a new `KrylovMethod::Lsmr` and dispatched inline from `Solver::solve`.
  Avoids explicit normal-equation formation for improved numerical
  conditioning.

### Changed

- `SolveResult.iterations` and `BatchSolveResult.iterations` now report the
  total Krylov iterations across the initial solve and any iterative-refinement
  correction solves (previously: outer solve only).

### Fixed

- CG stagnation guard threshold tightened from `EPS * rz_init` to
  `EPS^2 * rz_init`. The old threshold fired at `||r||/||b|| ~ sqrt(EPS)`,
  colliding with user tolerances near `1e-8` and causing spurious
  non-convergence on well-conditioned problems. The new threshold fires
  only at the true numerical-noise floor `||r|| ~ EPS * ||b||`.
- Weighted-operator length checks are active in release builds.
  `Gramian::build_weighted`, `WeightedGramianOperator::new`, and
  `WeightedDesignOperator::new` now use `assert_eq!` (was
  `debug_assert_eq!`) on `weights.len() == design.n_rows`, restoring the
  always-on validation boundary previously held by
  `ObservationWeights::validate_for`. Previously, a mismatched weights
  slice could either panic on short input or silently mis-index in
  release builds.

## [0.1.0] - 2026-03-12

Initial release of `within`, a high-performance fixed-effects solver for
econometric panel data.

### Added

- **Iterative solvers:** Left-preconditioned CG and right-preconditioned
  GMRES(m) with restarts, stagnation detection, and lucky breakdown handling.
- **Schwarz preconditioners:** Additive (CG-compatible, symmetric) and
  multiplicative (sequential sweep with sparse residual update). Additive
  variant auto-selects between atomic scatter and parallel reduction strategies.
- **Domain decomposition:** Bipartite factor-pair subdomains with connected
  component splitting and partition-of-unity weights.
- **Schur complement reduction:** Exact dense path for small subdomains,
  exact sparse path, and approximate path via GKS clique-tree spectral
  sparsification.
- **Approximate Cholesky local solver** via the `approx-chol` crate, with
  block elimination exploiting bipartite Gramian structure.
- **Dual operator representations:** Explicit CSR Gramian (fused sortless
  assembly from pair blocks) and implicit D^T W D (three-pass, no matrix
  stored).
- **Iterative refinement** with adaptive inner tolerance for observation-space
  accuracy.
- **Batch solve** with Rayon parallelism over RHS vectors, sharing the
  precomputed preconditioner.
- **Persistent `Solver` class** for amortizing preconditioner construction
  across multiple solves.
- **Weighted least squares** support via observation weights.
- **Preconditioner serialization** via `postcard` for Python pickle support.
- **Python API** via PyO3/maturin: `solve()`, `solve_batch()`, `Solver`,
  with full type stubs and GIL release during computation.
- **Zero-copy Python boundary** for F-contiguous category arrays and
  contiguous response vectors.
- **Three Rust crates:** `schwarz-precond` (generic, reusable),
  `within` (FE domain), `within-py` (thin PyO3 bridge).
- Benchmark infrastructure: Criterion micro-benchmarks (Rust) and 18-suite
  Python benchmark framework with setup/solve/accuracy measurement.
- CI/CD: Multi-platform testing (Linux, macOS, Windows), clippy with
  `-D warnings`, `#![deny(missing_docs)]` on library crates, and
  multi-architecture wheel builds with build attestation.

### Performance

- Adaptive additive Schwarz reduction scheduling with `Auto`, `AtomicScatter`,
  and `ParallelReduction` backends.
- Worker-local reusable reduction buffers for additive parallel reduction.
- Fused Schur right-hand-side assembly and related reduced-system kernel
  cleanup.

### Fixed

- Nested Rayon deadlocks in additive `ParallelReduction` when local solves
  spawn inner parallel work.
