# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **BREAKING:** The serialized `Preconditioner` wire format changed (v13 → v14) to retain its complete construction config and configured reduction strategy; older additive payloads no longer decode.
- Rust `Preconditioner` objects expose their normalized construction configuration through `Preconditioner::config()`.
- Serialized `schwarz_precond::SchwarzPreconditioner` values now preserve the configured reduction strategy.
- **BREAKING:** `schwarz_precond::mlsmr` takes an `MlsmrOptions` in place of its trailing `local_size`.
- **BREAKING:** `LsmrStopReason` gains `Escalated` and `WarmStartExact`, breaking exhaustive `match`es.
- A warm start that already solves the system reports `WarmStartExact` instead of `ZeroRhs`.
- **BREAKING:** The serialized `Preconditioner` wire format changed (v12 → v13) with the `approx-chol` 0.4 → 0.5 bump; 0.3.0 bytes no longer decode.

### Added

- `schwarz_precond::EscalationPolicy` builds a per-run `EscalationHandler` that ends a solve with `LsmrStopReason::Escalated` and an iterate that warm-starts the next preconditioner; `Staleness` implements it from the trailing contraction window.
- `schwarz_precond::MlsmrOptions::warm_start` carries an initial iterate through a change of preconditioner.

### Fixed

- A design carrying varying slopes on two distinct factors could fail preconditioner construction with `matrix is not symmetric`, when rounding left the two triangles of the exact Schur complement unequal (#229).
- A `design` that is neither a 2-D `uint32` array nor a list of `Effect` raised `ValueError` where the documented type is `TypeError`, and `AdditiveSchwarz` accepted a wrong-type `local_solver` at construction, deferring the `TypeError` to solve time (#248).

### Removed

- `faer` is no longer a dependency of `schwarz-precond`; it is used only by the `custom_local_solver` example and moves to dev-dependencies.

## [0.3.0] - 2026-07-30

### Added

- **Varying slopes:** the `Effect` term type (level codes, an intercept flag, zero or more slope columns) is accepted anywhere a categories matrix is, in Rust and Python (#58–#63).
- `SolveResult` / `BatchSolveResult` report unidentified directions in `unidentified` — Rust `CoefficientAddress`, Python `UnidentifiedDirection`; those coefficient slots hold `0`, never NaN (#69).
- `layout` (`CoefficientLayout`) converts between a coefficient address and its flat `x` index (#99).
- `SolveResult.x` is term-major: coefficient column `c` of `level` sits at `term_offset + c * n_levels + level`. Intercept-only designs keep the 0.2.0 ordering (#71).
- `BatchSolveResult.time_setup` reports the shared per-batch setup time (#194).
- `ScalingConfig` tunes signed-component scaling certification; `Solver::warnings()` returns non-fatal `BuildWarning`s (#61).
- `BuildError::{EmptyEffect, SlopeLengthMismatch, InvalidLoading, UnscalableComponent, DofSpaceExceedsU32}` (#58, #125, #160).
- A `py.typed` marker, so type checkers pick up the shipped stubs (#142).

### Changed

- **BREAKING:** The Python `solve` / `solve_batch` / `Solver` first parameter is renamed `categories` → `design` (#58).
- **BREAKING:** The Python free `solve` / `solve_batch` take `(design, y, weights, options, preconditioner)`; `options` must now be passed by keyword (#101).
- **BREAKING:** The serialized `Preconditioner` wire format changed (v3 → v12); 0.2.0 bytes no longer decode (#72, #98).
- **BREAKING:** `LocalSolverConfig.approx_schur` becomes `schur: SchurMode` (Rust) / `Schur | None` (Python), where `None` means the default (approximate) and exact is `SchurMode::Exact` / `Schur.exact()`. The `within.config.LocalSolverConfig` compatibility subclass is removed, so `LocalSolverConfig` is no longer picklable (#104).
- **BREAKING:** `SolveResult` / `BatchSolveResult` gain `unidentified` and `layout`, breaking struct literals and exhaustive destructuring (#69, #99).
- **BREAKING:** `Design<'a>` / `Solver<'a>` take a lifetime instead of a storage type parameter, borrowing caller columns until a locality sort or `into_owned()` (#68).
- **BREAKING:** `Design::from_store` → `Design::from_frame`, taking an `ObservationFrame` — columnar storage with each column independently borrowed or owned (#68).
- **BREAKING:** `BuildError::ObservationCountMismatch` renames `factor` → `column`; `SingularDiagonal` drops `block` (#68).
- **BREAKING:** `ndarray` 0.16 → 0.17, which appears in the public API as `IntoDesign for ArrayView2<u32>`.
- **BREAKING:** The minimum supported Rust version is 1.85 (#123).
- Python wheels are stable-ABI: one `cp39-abi3` wheel per platform runs on any CPython ≥ 3.9 (#161).
- `approx-chol` 0.2.0 → 0.4.0, speeding up local-solver setup.
- `Design` locality-sorts unsorted observations; results return in caller row order (#68). Together with the new `approx-chol` sampler, this makes coefficients agree with 0.2.0 within solver tolerance rather than bitwise.
- `SolveResult.residual` reports the LSMR recurrence's normal-equation residual estimate instead of recomputing it exactly (#149).
- Rust solve entry points take LSMR options as `impl Into<Option<&LsmrOptions>>`, so `None` selects the default (#105).
- Python raises `RuntimeError` for runtime failures and reserves `ValueError` for invalid input (#179, #180), and `TypeError` naming the expected dtype for wrong-dtype arrays (#100).
- The one-shot Python `solve` / `solve_batch` re-emit build warnings as `UserWarning`s, matching the persistent `Solver` (#103).

### Fixed

- A response with entries above ~1e154 overflowed the LSMR initial norm, silently returning `x = 0` as converged (#164).
- A near-breakdown solve was aborted as a non-positive-definite preconditioner when the M-norm rounded slightly negative (#118).
- Deserializing untrusted `Preconditioner` bytes could panic or scatter out of bounds; the invariants are now validated (#155, #166).
- Python `Preconditioner.apply` releases the GIL around the native apply (#180).
- Python `solve_batch` emitted the F-contiguity `UserWarning` twice.

### Removed

- `scipy` is no longer a runtime dependency of `within-py` — the package never imported it.
- **BREAKING:** The `Store` trait and its `ArrayStore` / `FactorMajorStore` backends, superseded by `ObservationFrame` (#68).
- **BREAKING:** `schwarz_precond::LsmrStopReason::BidiagonalizationBreakdown`; the tolerance test now fires on the same step.

## [0.2.0] - 2026-06-04

Modified LSMR is now the sole iterative solver, replacing CG and GMRES.

### Added

- **Modified LSMR:** preconditioned `mlsmr` on `sqrt(W) D` (no normal-equation formation), optional windowed mGS reorthogonalization via `LsmrOptions.local_size`, and rejection of non-finite input with `SolveError::InvalidInput` (was silent NaN propagation).
- `PreconditionerConfig::Off` variant — explicit identity preconditioner.
- `PreconditionerConfig::Diagonal` variant — diagonal/Jacobi preconditioner (`M⁻¹ = diag(DᵀWD)⁻¹`), exposed in Python as `PreconditionerConfig.Diagonal`. A zero diagonal (an unobserved or fully zero-weighted level — an unidentified DOF) takes the pseudo-inverse (`inv = 0`), pinning that coordinate to 0 like the unpreconditioned path; only a non-finite reciprocal is rejected with `BuildError::SingularDiagonal`.
- Python `within.config` submodule: `AdditiveSchwarz`, `LocalSolverConfig`, `ApproxCholConfig`, `ApproxSchurConfig`, `ReductionStrategy`.
- Python `Solver` / `solve` / `solve_batch` accept a 5-form preconditioner: `None`, `PreconditionerConfig.{Off, Additive, Diagonal}`, `AdditiveSchwarz(...)`, or a pre-built `Preconditioner` (reuse path).
- `From<&Preconditioner> for PreconditionerInput`: `Solver::new(.., &precond)` now works alongside the owned form. Cloning a `Preconditioner` is O(1) (refcount-only), so this is a cheap reuse path.
- `BuildError::PreconditionerDimensionMismatch { expected, actual_rows, actual_cols }`: `Solver::new` fails fast when a reused preconditioner's shape does not match the design's DOF count, instead of bubbling up an opaque error from inside the iterative solver.

### Changed

- **BREAKING:** Renamed (Rust + Python): `SolverParams` → `LsmrOptions`, `Preconditioner` config enum → `PreconditionerConfig`, `FePreconditioner` → `Preconditioner`, `SchurComplement` → `LocalSolverConfig`.
- **BREAKING:** `Preconditioner` is an opaque struct (was an enum); `#[non_exhaustive]` removed; `#[serde(transparent)]` pins the wire format (future variants must be append-only).
- **BREAKING:** The internal `CrossTab` no longer stores or serializes the factor-pair diagonal blocks (`D_q`, `D_r`) — they are build-time-only inputs, folded into the local factor and dropped after build. This shrinks the serialized `Preconditioner` payload (wire-format fixture bumped v2 → v3).
- **BREAKING:** `within::{domain, operator, orchestrate, solver}` are `pub(crate)`; public items remain re-exported from the crate root.
- **BREAKING:** `LocalSolverConfig::default()` uses `split_merge: Some(2)` (was structural zero).
- **BREAKING:** `Solver::solve` / `solve_batch` reject `y.len() != n_obs` / `Y.shape[0] != n_obs` (was silent truncation in weighted mode).
- **BREAKING:** `PreconditionerConfig` and `schwarz_precond::SolveError` are `#[non_exhaustive]`; `SolveError` gains `InvalidInput { context, message }`.
- **BREAKING:** `LocalSolver::solve_local` takes `allow_inner_parallelism: bool`. `SchwarzPreconditioner` drops its `I: LocalSolveInvoker` type parameter.
- **BREAKING:** `Operator::apply` / `apply_adjoint` return `Result<(), SolveError>`; `ApplyError` removed (variants moved onto `SolveError`). Python `Preconditioner.apply` raises `RuntimeError` instead of returning NaNs (#29).
- **BREAKING:** Error vocabulary collapsed to per-crate `BuildError` / `SolveError`. `LocalSolveError::ApproxCholSolveFailed` → `BackendFailed`; `LocalSolveError` is `#[non_exhaustive]` (#30).
- **BREAKING:** `ObservationStore` → `Store`; `WeightedDesign` → `Design`; `WeightedDesignOperator` → `DesignOperator` (#28).
- **BREAKING:** Observation weights externalized from the store layer. `FactorMajorStore::new` / `ArrayStore::new` drop their weights argument; `Solver::new` and `build_preconditioner` gain weights parameters.
- **BREAKING:** `Solver::new` reshaped to take `impl IntoDesign` + `impl Into<PreconditionerInput>`, accepting raw categories (`ArrayView2<u32>`) or a pre-built `Design`, and any of `None` / `&PreconditionerConfig` / `Some(&PreconditionerConfig)` / owned `PreconditionerConfig` / owned-or-borrowed `Preconditioner`. `LsmrOptions` moved off the constructor onto `Solver::solve` / `solve_batch` (Rust) and `solver.solve(y, options=...)` / `solver.solve_batch(Y, options=...)` (Python); the persistent `Solver` now owns only problem state (design, weights, preconditioner). The legacy `Solver::from_design`, `Solver::from_design_with_preconditioner`, and `Solver::with_preconditioner` constructors are removed; the free `solve` / `solve_batch` Python kwarg was renamed `config=` → `options=`.
- **BREAKING:** `Design` is pure data + layout — `matvec_d`, `rmatvec_dt`, `rmatvec_wdt`, `gramian_diagonal`, `uid_weight` removed; use `DesignOperator::new(&design, weights)` instead.
- `DesignOperator::new` validates `weights.len() == design.n_rows`; weighted `apply` / `apply_adjoint` no longer allocate.
- `build_preconditioner` returns `BuildError::WeightCountMismatch` for wrong-length weights (was OOB panic in `CrossTab`).
- Python: `LocalSolverConfig(approx_schur=None)` requests exact Schur; omitting uses the library default approximate.
- Python: all PyO3 classes report `__module__ == "within._within"` (was `"builtins"`).
- LSMR vector kernels parallelized via Rayon.
- **BREAKING:** `BatchSolveResult` fields are now `pub` (was `pub(crate)` behind accessor methods); only `x(i)` and `demeaned(i)` slicing methods remain.
- **BREAKING:** `Design::n_rows` renamed to `n_obs`; new `Design::n_obs()` / `n_dofs()` accessors.
- **BREAKING:** `Design` fields are `pub(crate)`.
- **BREAKING:** `DesignOperator` is `pub(crate)`.
- **BREAKING:** `SolveResult.final_residual` / `BatchSolveResult.final_residual` renamed to `residual` (Rust + Python).
- **BREAKING:** `SchwarzPreconditioner::new(entries, strategy)` replaces `new(entries, n_dofs)` / `with_strategy(entries, n_dofs, strategy)`; `n_dofs` derived from entries. `resolved_reduction_strategy()` renamed to `reduction_strategy()`; dead `with_reduction_strategy` + configured getter removed. `BuildError::GlobalIndexOutOfBounds` removed.
- **BREAKING (Python):** `ApproxCholConfig.split: int (1=off)` → `split_merge: int | None (None=off)`, matching Rust. Pickle payload shape changed.
- **BREAKING:** weights types now mirror each API's relationship to the data: the persistent `Solver::new` takes owned `weights: Option<Vec<f64>>` (it holds them across solves), while the one-shot `solve` / `solve_batch` take borrowed `weights: Option<&[f64]>`. A bare `None` works for both (no turbofish). `WithinError` is `#[non_exhaustive]`.
- Free `solve()` / `solve_batch()` accept `impl Into<PreconditionerInput>` (same shapes as `Solver::new`).
- `Solver` and `Preconditioner` implement `Debug`.
- `approx-chol` bumped `0.1` → `0.2` (now published on crates.io); the new upstream sampler may produce slightly different fill edges in the Schur complement.

### Fixed

- `SchwarzPreconditioner::apply` rejects `r.len() != n_dofs` or `z.len() != n_dofs` with `SolveError::InvalidInput` (was an unchecked out-of-bounds path).
- `BufferPool` no longer recycles a partially-written atomic-scatter buffer when the preceding apply returned an error — the buffer is dropped so the next caller starts from a freshly-zeroed allocation rather than inheriting stale atomic state.
- Python `Solver.solve_batch` accepts `Y` as a keyword argument again — the Rust impl exposed it as `y_matrix=` while the stub advertised `Y=`, so `solver.solve_batch(Y=...)` failed at runtime.
- Python `Solver.solve_batch` now validates `Y.shape[0]` against the solver's `n_obs` up front (parity with the free `solve_batch`); previously an empty batch with the wrong row count silently returned a success.
- `ArrayStore::factor_column` falls back to safe per-element access for negative-column-stride numpy views (e.g. `cats[:, ::-1]`), closing an out-of-bounds read reachable from Python.
- CSR index construction (Schur + cross-tab) uses checked `usize`→`u32` conversions that panic above `u32::MAX` nonzeros instead of silently truncating.
- Negative or non-finite observation weights are rejected up front with `BuildError::InvalidWeight { index, value }` — the operator applies `W^{1/2}`, so a bad weight previously took `sqrt` of a NaN/negative and silently corrupted the solution.
- The Schur fill-edge reduction is order-independent: duplicate edge weights are summed in a total `(lo, hi, weight)` order, so the assembled Schur complement is bit-for-bit reproducible across runs and thread counts (parallel summation order no longer depends on thread scheduling).
- Modified LSMR returns `SolveError::InvalidInput` for a non-positive-definite preconditioner (`⟨v, Mv⟩ < 0`) instead of silently converging to a wrong solution.
- Additive-Schwarz `apply` preserves the original solve error (no longer masked by a buffer-pool error) and zeroes its output on the reduction error path.
- Python `solve_batch` raises `ValueError` instead of an opaque `PanicException` on an internal shape-invariant violation.

### Removed

- **BREAKING:** `schwarz_precond::SparseMatrix` removed from the public surface (it was renamed `CsrMatrix` earlier this cycle), along with its `From<faer::SparseRowMatRef>` conversion. The reduced Schur / Laplacian CSR representation is now internal to `within`'s `block_elim` module; `schwarz-precond`'s public API narrows to its `Operator` / `LocalSolver` traits and solvers (#52).
- **BREAKING:** Top-level Rust re-exports: `within::Operator` (use `schwarz_precond::Operator`), `within::DEFAULT_DENSE_SCHUR_THRESHOLD`, `within::Subdomain`, `within::domain::{PartitionWeights, SubdomainCore}`.
- **BREAKING:** `LocalSolverConfig::solver_default()` (the single `default()` now serves both paths).
- **BREAKING:** CG, GMRES, multiplicative Schwarz, iterative refinement, and support types: `KrylovMethod`, `OperatorRepr`, `Multiplicative` variants, `SolverParams.max_refinements`, Python `CG` / `GMRES` / `MultiplicativeSchwarz`, `ResidualUpdater`, `OperatorResidualUpdater`, `IdentityOperator`.
- **BREAKING:** `Gramian`, `GramianOperator`, the previous bare `DesignOperator`, `build_schwarz`, `FeSchwarz`, `WithinError::Overflow` from the `within` public surface.
- **BREAKING (Rust + Python):** `Preconditioner::n_subdomains` and `Preconditioner::subdomain_inner_parallel_work` (and their Python `@property` counterparts) — internal diagnostics, not part of the stable surface.
- **BREAKING:** `schwarz_precond::solve::{cg, gmres}` and the `solve` module; `schwarz_precond::schwarz::{additive, multiplicative}` flattened into `schwarz_precond::schwarz`.
- **BREAKING:** `AdditiveSchwarzDiagnostics` and related accessors (#34).
- **BREAKING:** `ObservationWeights` enum; `Store::weight` and `Store::is_unweighted` (#28).
- **BREAKING:** `WithinResult` type alias.

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
