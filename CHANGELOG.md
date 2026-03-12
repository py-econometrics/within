# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-12

### Added

- Initial release of the Rust `within` and `schwarz-precond` crates and the Python `within` package.
- Fixed-effects normal-equation solvers with conjugate gradient and right-preconditioned GMRES.
- Additive and multiplicative one-level Schwarz preconditioners.
- Local solver support for full SDDM solves and Schur-complement-based solves backed by approximate Cholesky factorizations.
- Python bindings for solving weighted and unweighted fixed-effects problems from categorical panel data.
- Benchmark and verification harnesses covering panel, fixest-style, high-FE, scaling, and pathological graph regimes.

### Performance

- Adaptive additive Schwarz reduction scheduling with `Auto`, `AtomicScatter`, and `ParallelReduction` backends.
- Worker-local reusable reduction buffers for additive parallel reduction.
- Fused Schur right-hand-side assembly and related reduced-system kernel cleanup.

### Fixed

- Nested Rayon deadlocks in additive `ParallelReduction` when local solves spawn inner parallel work.
- Regression coverage for both the generic nested-Rayons deadlock shape and the real `within` `BlockElimSolver` path.
