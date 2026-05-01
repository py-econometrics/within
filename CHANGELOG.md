# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

> Breaking changes affect Rust consumers of `within` and `schwarz-precond`
> only. The Python API (`within.solve`, `within.Solver`) is unchanged.

### Breaking Changes

- **Error enums consolidated from six to two.** `SubdomainCoreBuildError`,
  `SubdomainEntryBuildError`, `PreconditionerBuildError` → `BuildError`.
  `LocalSolveError`, `ApplyError`, and the prior `SolveError` →
  `SolveError` (`#[non_exhaustive]`) with variants `LocalSolveFailed`,
  `Synchronization`, `InvalidInput`. Match arms need a wildcard.
- **`Operator::apply` / `apply_adjoint` are now fallible**, returning
  `Result<(), SolveError>`. `try_apply` / `try_apply_adjoint` removed;
  propagate with `?` at call sites.
- **`LocalSolveInvoker` removed; parallelism hint moved to `LocalSolver`.**
  `solve_local` now takes `allow_inner_parallelism: bool`.
  `SchwarzPreconditioner` / `MultiplicativeSchwarzPreconditioner` drop
  their `I: LocalSolveInvoker` type parameter. `with_strategy_and_invoker`
  → `with_strategy`.
- **Krylov solvers split back into idiomatic two-function pairs.** `pcg` /
  `pgmres` / `mlsmr` no longer take `Option<&M>`; they require `&M` and
  are paired with `cg` / `gmres` / `lsmr` for the unpreconditioned case.
  This restores the std-library convention (cf. `Vec::new` /
  `Vec::with_capacity`) and removes the `None::<&SomeType>` turbofish
  burden at every unpreconditioned call site. `cg_solve`,
  `cg_solve_preconditioned`, `gmres_solve`, and `preconditioned_lsmr`
  remain removed (their replacements are the new `cg` / `pcg` etc.).
- **`FeSchwarz` is now a type alias** of
  `SchwarzPreconditioner<BlockElimSolver>`; the newtype wrapper and its
  delegation methods are gone — call methods directly on the alias.
- **`build_schwarz` removed.** Use
  `build_preconditioner(&design, weights, gramian, &Preconditioner::Additive(cfg, ReductionStrategy::default()))`.
- **Additive-Schwarz diagnostic free functions are now inherent methods**
  on `FePreconditioner`: `additive_reduction_strategy`,
  `resolved_additive_reduction_strategy`, `additive_schwarz_diagnostics`.
  Call as `preconditioner.additive_schwarz_diagnostics()` etc.
- **`FePreconditioner::subdomain_inner_parallel_work` removed** (Rust and
  Python). The per-subdomain work vector had no in-tree callers; the
  actionable scheduling signals (`total_inner_parallel_work`,
  `max_inner_parallel_work`, `outer_parallel_capacity`) remain available
  on `AdditiveSchwarzDiagnostics`.
- **`schwarz_precond::IdentityOperator` removed.** Its original purpose
  (deduplicating CG/LSMR by passing identity as preconditioner) was made
  obsolete by the `pcg`/`pgmres`/`mlsmr` refactor to `Option<&M>`. Tests
  that needed turbofish disambiguation can use any concrete `Operator`
  type already in scope (`None::<&MyOp>`).
- **Weights externalized from the store/design layer.** `ObservationStore`
  → `Store` (no `weight()` / `is_unweighted()`); `ObservationWeights`
  deleted. `FactorMajorStore::new` / `ArrayStore::new` no longer take
  weights. `WeightedDesign` → `Design` (pure data + metadata); the
  `matvec_d`, `rmatvec_dt`, `rmatvec_wdt`, `gather_add`, `scatter_add`,
  `uid_weight`, `is_unweighted`, `gramian_diagonal` methods are gone —
  use `DesignOperator` / `Gramian::build(&design, None|Some(&w))`.
- **Operator layer carries optional weights on a single struct per role.**
  `GramianOperator` covers both `D^T D` and `D^T W D`; `DesignOperator`
  covers both `D` and `sqrt(W) D` (the LSMR rectangular form). One
  constructor each: `GramianOperator::new(&design, weights: Option<&[f64]>)`
  and `DesignOperator::new(&design, weights: Option<&[f64]>)` —
  pass `None` for unweighted, `Some(&w)` for weighted. The branch on
  weights is hoisted outside the per-row hot loop, so each variant
  still monomorphizes to its own scatter kernel. Explicit assembly:
  `Gramian::build(&design, weights: Option<&[f64]>)` (replaces both
  prior `build` and `build_weighted`). `GramianOperator` borrows
  `&'a [f64]` weights with the design's lifetime; `DesignOperator`
  owns `Vec<f64>` of precomputed `sqrt(W)`.
- **`Solver::from_design{,_with_preconditioner}` take
  `weights: Option<Vec<f64>>`** between `design` and `params`.
  `Solver::with_preconditioner` removed — use
  `from_design_with_preconditioner` directly.
- **Internal builders (`build_schwarz`, `build_preconditioner`,
  `CrossTab::build_for_pair`, `domain::factor_pairs`) take
  `Option<&[f64]>` weights** explicitly.
- **`ArrayStore::new` is infallible** — returns `Self`, not
  `WithinResult<Self>`. Drop `?` / `.unwrap()` at call sites.
- **`MultiplicativeSchwarzPreconditioner::new`** drops the
  `symmetric: bool` parameter (production always passed `false`).
  Serialized blobs now have one fewer field; older blobs containing
  `symmetric` won't deserialize.
- **`within::observation::validate_weights` removed** (inlined at its
  sole call site).
- **`IdentityOperator`** no longer used by `pcg`/`pgmres` (they inline
  `copy_from_slice` for the unpreconditioned path); type remains `pub`
  for tests/examples.

### Added

- **LSMR rectangular least-squares solver** — preconditioned LSMR on
  the weighted design operator (`sqrt(W) D`), exposed as
  `KrylovMethod::Lsmr`. Avoids explicit normal-equation formation.

### Changed

- `SolveResult.iterations` / `BatchSolveResult.iterations` now report
  total Krylov iterations across the initial solve plus any
  iterative-refinement corrections (previously: outer solve only).

### Fixed

- **CG stagnation guard tightened** from `EPS * rz_init` to
  `EPS^2 * rz_init`. The old threshold fired at `~sqrt(EPS)`, causing
  spurious non-convergence near user tolerances of `1e-8`.
- **Weighted-operator length checks active in release builds.**
  `Gramian::build`, `GramianOperator::new`, and `DesignOperator::new`
  (each when called with `Some(&w)`) now use `assert_eq!` (was
  `debug_assert_eq!`) for `weights.len() == design.n_rows`. A mismatch
  could previously mis-index silently in release.

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
