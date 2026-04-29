# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking Changes

> All items in this section break source compatibility for Rust consumers
> of the `within` and `schwarz-precond` crates. The Python boundary
> (`within.solve`, `within.Solver`, type stubs) is unchanged.

- **Error enums consolidated from six to two.**
  `SubdomainCoreBuildError`, `SubdomainEntryBuildError`, and
  `PreconditionerBuildError` collapse into a single `BuildError` carrying
  all four prior variants. `LocalSolveError`, `ApplyError`, and the
  previous `SolveError` collapse into a single `SolveError` (still
  `#[non_exhaustive]`) with three variants:
  `LocalSolveFailed { subdomain, context, message }`,
  `Synchronization { context }`, and `InvalidInput { context, message }`.
  `SolveError::Apply(ApplyError)` and `ApplyError::LocalSolveFailed { source }`
  no longer exist; the chained `From<ApplyError>` conversion is gone.
  Local solvers construct `SolveError::LocalSolveFailed` with a placeholder
  subdomain index — the Schwarz exec loop re-tags via an internal
  `tag_subdomain` helper. Downstream consumers matching on `SolveError`
  must include a wildcard arm; future variant additions will not be breaking.
- **`Operator` trait collapsed to a single fallible `apply`.** `apply` and
  `apply_adjoint` now return `Result<(), SolveError>` directly; the
  `try_apply` / `try_apply_adjoint` methods are removed. Every `op.apply(x, y)`
  call site must propagate (`?`) or `.expect(...)`. The Schwarz preconditioner
  NaN-fill-on-error workaround in `apply` is gone — failures surface as
  `Err(SolveError::LocalSolveFailed { .. })`.
- **`LocalSolveInvoker` trait removed; parallelism hint folded into
  `LocalSolver`.** `LocalSolveInvoker`, `DefaultLocalSolveInvoker`, and
  `FeLocalSolveInvoker` are deleted. `LocalSolver::solve_local` now takes
  `allow_inner_parallelism: bool` directly:

  ```rust
  fn solve_local(&self, rhs: &mut [f64], sol: &mut [f64],
                 allow_inner_parallelism: bool) -> Result<(), SolveError>;
  ```

  `SchwarzPreconditioner` and `MultiplicativeSchwarzPreconditioner` lose
  their second type parameter (`I: LocalSolveInvoker<S>`).
  `SchwarzPreconditioner::with_strategy_and_invoker` is replaced by
  `with_strategy`. Solvers that don't care about the hint should accept
  the parameter and ignore it.
- **Public CG/GMRES entry points reduced to `pcg` and `pgmres`.** The
  layered `cg_solve`, `cg_solve_preconditioned`, and `gmres_solve` entry
  points are gone (merged into `pcg`/`pgmres` bodies, which dispatch on
  `Option<&M>` directly). External callers pass `None::<&M>` for
  unpreconditioned solves.
- **`Solver::with_preconditioner` removed.** The two-line
  `ArrayStore`/`Design` construction is now inlined at call sites; use
  `Solver::from_design_with_preconditioner` directly.
- **`FeSchwarz` is now a type alias** of
  `SchwarzPreconditioner<BlockElimSolver>`. The newtype wrapper, its seven
  delegation methods (`subdomains`, `reduction_strategy`,
  `resolved_reduction_strategy`, `diagnostics`, `with_reduction_strategy`,
  etc.), and the `impl Operator` forwarding block are deleted. Callers
  invoke methods on the alias directly.
- **`IdentityOperator` is no longer used in production paths.** `pcg`/`pgmres`
  inline the unpreconditioned dispatch (`copy_from_slice` instead of
  applying the identity). The type stays `pub` for tests/examples that
  need an identity preconditioner instance.
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
  - `design.gramian_diagonal()` → `Gramian::build(&design, None).diagonal()`
    (or `Gramian::build(&design, Some(&w)).diagonal()` when weighted).
- **Operator layer split into weighted / unweighted variants.** The
  implicit Gramian splits into `GramianOperator` (`D^T D`) and the new
  `WeightedGramianOperator` (`D^T W D`); the design operator splits into
  `DesignOperator` (`D`) and the new `WeightedDesignOperator` (`sqrt(W) D`,
  used by the LSMR path). Each Gramian operator exposes `to_csr()`.
  Explicit assembly is now `Gramian::build(&design, weights)` taking
  `weights: Option<&[f64]>`, which dispatches on the optional slice.
  Weighted apply / apply\_adjoint are single fused passes — the
  `Mutex` scratch buffer is gone.
- **`Solver::from_design` and `Solver::from_design_with_preconditioner`
  take an explicit `weights: Option<Vec<f64>>` argument** between
  `design` and `params`, mirroring the externalized weights. Passing
  `None` selects the unweighted operator pair at solve time.
- **Internal builders take `Option<&[f64]>` weights:** `build_schwarz`,
  `build_preconditioner`, `CrossTab::build_for_pair`, and the
  `domain::factor_pairs` constructors. Callers using these lower-level
  entry points directly must now pass weights explicitly.
- **`schwarz-precond`: `lsmr` and `preconditioned_lsmr` removed.** Both
  bodies are folded into `mlsmr`, which now dispatches on its
  `preconditioner: Option<&M>` argument. Callers pass `None::<&M>` for
  the unpreconditioned path and `Some(&m)` for the preconditioned path.
- **`ArrayStore::new` no longer returns `WithinResult`.** The signature
  becomes `pub fn new(categories: ArrayView2<'a, u32>) -> Self`. The
  prior `Result` was vestigial (the body was `Ok(Self { categories })`);
  drop the `?` / `.unwrap()` at call sites.
- **`Gramian::build` and `Gramian::build_weighted` unified.** Merged into
  a single `Gramian::build(design, weights: Option<&[f64]>)` that
  dispatches on the optional weight slice. The standalone
  `Gramian::build_weighted` is removed: pass `Some(&w)` to `Gramian::build`
  for the weighted form, `None` for the unweighted form.
- **`MultiplicativeSchwarzPreconditioner::new` signature change.** The
  `symmetric: bool` parameter is removed; the forward+backward symmetric
  sweep (and the corresponding `if self.symmetric { .. }` branch in
  `apply`) is gone. Production code always passed `false`. Serialized
  preconditioners (postcard / serde) carry one fewer field — the
  `serialize_struct` field count drops 4 -> 3, and `Helper` no longer
  has a `symmetric` field. Previously serialized blobs that include the
  `symmetric` field will fail to deserialize.
- **`WeightedGramianOperator::new` borrows weights.** Signature becomes
  `pub fn new(design: &'a Design<S>, weights: &'a [f64]) -> Self` —
  the operator now borrows the weight slice with the same lifetime as
  the design reference, instead of cloning into an owned `Vec<f64>`.
  Callers must ensure the weight slice outlives the operator.
  Eliminates a per-construction `weights.to_vec()` clone on every
  iterative-refinement step.
- **`within::observation::validate_weights` removed.** The four-line
  `if w.len() != n_obs { Err(WeightCountMismatch { .. }) }` check is
  inlined at its single call site (`Solver::from_design_with_source`).
  External callers that imported the helper should inline the same
  guard locally.

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
- Schwarz preconditioner generic surface narrowed: `SchwarzPreconditioner<S>`
  and `MultiplicativeSchwarzPreconditioner<S, U>` (was `<S, I>` /
  `<S, U, I>`). The `I: LocalSolveInvoker<S>` parameter is gone; the
  parallelism hint is now part of the `LocalSolver` contract.

### Removed

- `LocalSolveInvoker` trait, `DefaultLocalSolveInvoker`, and the within-crate
  `FeLocalSolveInvoker` wrapper. Use `LocalSolver::solve_local`'s new
  `allow_inner_parallelism: bool` parameter instead.
- `Operator::try_apply` and `Operator::try_apply_adjoint`. The fallible
  `apply` / `apply_adjoint` are now the only methods.
- `cg_solve`, `cg_solve_preconditioned`, and `gmres_solve` public entry
  points. Use `pcg` and `pgmres` (which take `Option<&M>` for the
  preconditioner).
- `Solver::with_preconditioner`. Use `Solver::from_design_with_preconditioner`
  after constructing an `ArrayStore` + `Design` explicitly.
- `SchwarzPreconditioner::with_strategy_and_invoker`. Use `with_strategy`
  (the parallelism hint flows through `LocalSolver::solve_local` instead).
- `FeSchwarz` newtype wrapper and its delegation methods. `FeSchwarz` is
  now a type alias of `SchwarzPreconditioner<BlockElimSolver>`; methods
  resolve directly on the alias.
- `SubdomainCoreBuildError`, `SubdomainEntryBuildError`,
  `PreconditionerBuildError`, `LocalSolveError`, `ApplyError` enums. All
  variants now live on `BuildError` or `SolveError`.
- `lsmr` and `preconditioned_lsmr` public entry points in
  `schwarz-precond`. Use `mlsmr` with `Option<&M>` for the preconditioner.
- `Gramian::build_weighted`. Folded into `Gramian::build(design, Option<&[f64]>)`.
- `within::observation::validate_weights`. Inlined at its sole call site
  in `Solver` construction.
- `MultiplicativeSchwarzPreconditioner::new`'s `symmetric: bool`
  parameter and the symmetric-sweep branch in `apply`. Production
  code always passed `false`.

### Fixed

- CG stagnation guard threshold tightened from `EPS * rz_init` to
  `EPS^2 * rz_init`. The old threshold fired at `||r||/||b|| ~ sqrt(EPS)`,
  colliding with user tolerances near `1e-8` and causing spurious
  non-convergence on well-conditioned problems. The new threshold fires
  only at the true numerical-noise floor `||r|| ~ EPS * ||b||`.
- Weighted-operator length checks are active in release builds.
  `Gramian::build` (when called with `Some(&w)`),
  `WeightedGramianOperator::new`, and `WeightedDesignOperator::new`
  now use `assert_eq!` (was `debug_assert_eq!`) on
  `weights.len() == design.n_rows`, restoring the always-on validation
  boundary previously held by `ObservationWeights::validate_for`.
  Previously, a mismatched weights slice could either panic on short
  input or silently mis-index in release builds.

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
