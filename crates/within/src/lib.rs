#![deny(missing_docs)]
//! Fixed-effects normal equation solver with Schwarz preconditioning.
//!
//! `within` solves the linear fixed-effects problem **y = D x + e** where D is a
//! sparse categorical design matrix (each row has exactly Q ones, one per factor).
//! The normal equations **G x = D^T W y** (G = D^T W D) are solved via
//! preconditioned CG or right-preconditioned GMRES with additive or
//! multiplicative Schwarz preconditioners backed by approximate Cholesky
//! local solvers.
//!
//! # Problem formulation
//!
//! ## The fixed-effects model
//!
//! Panel data regressions often include *fixed effects* — per-group intercepts
//! for categorical variables such as firm, worker, or year. The linear model is:
//!
//! ```text
//! y = D alpha + X beta + epsilon
//! ```
//!
//! where **D** is the design matrix of fixed effects (stacked indicator columns,
//! one per level of each factor), **alpha** are the fixed-effect coefficients,
//! **X** are additional covariates, and **beta** their coefficients.
//!
//! In practice, econometric packages "absorb" the fixed effects by projecting
//! out **D** (i.e., demeaning **y** and **X** within groups). This projection
//! requires solving the *normal equations* of the fixed-effects part alone.
//!
//! ## Normal equations and the Gramian
//!
//! The weighted least-squares projection solves:
//!
//! ```text
//! G alpha = D^T W y       where  G = D^T W D
//! ```
//!
//! Here **W** is a diagonal weight matrix (often the identity, but WLS and
//! iteratively reweighted least squares produce non-trivial weights). The
//! matrix **G** is the *Gramian* of the weighted design matrix.
//!
//! ## Block structure of the Gramian
//!
//! Because **D** is block-diagonal — `D = [D_1 | D_2 | ... | D_Q]` with one
//! indicator sub-matrix per factor — the Gramian inherits a natural Q x Q
//! block structure:
//!
//! ```text
//! G = [ D_1^T W D_1    D_1^T W D_2    ...  D_1^T W D_Q ]
//!     [ D_2^T W D_1    D_2^T W D_2    ...              ]
//!     [      ...             ...       ...              ]
//!     [ D_Q^T W D_1         ...        ... D_Q^T W D_Q ]
//! ```
//!
//! - **Diagonal blocks** `G_{qq} = D_q^T W D_q` are diagonal matrices whose
//!   entries are the weighted observation counts per level of factor *q*.
//! - **Off-diagonal blocks** `G_{qr} = D_q^T W D_r` are cross-tabulation
//!   matrices — entry `(i, j)` counts the weighted observations that belong
//!   to level *i* of factor *q* and level *j* of factor *r*.
//!
//! ## Why this is hard
//!
//! The dimension of **G** equals the total number of factor levels across all
//! factors. With millions of firms, workers, or products, **G** can easily
//! reach millions of rows. Direct factorization is infeasible. However, **G**
//! is extremely sparse — each observation contributes at most Q*(Q+1)/2
//! entries — so iterative Krylov solvers (CG, GMRES) with good
//! preconditioners are the natural approach. That is where domain
//! decomposition comes in: each factor pair defines a subdomain, and a
//! Schwarz preconditioner applies local approximate solves on these
//! overlapping subdomains.
//!
//! # Quick start
//!
//! ```no_run
//! use ndarray::Array2;
//! use within::{solve, SolverParams};
//!
//! // Two factors: 100 levels each, 10 000 observations
//! let n_obs = 10_000usize;
//! let mut categories = Array2::<u32>::zeros((n_obs, 2));
//! for i in 0..n_obs {
//!     categories[[i, 0]] = (i % 100) as u32;
//!     categories[[i, 1]] = (i / 100) as u32;
//! }
//! let y: Vec<f64> = (0..n_obs).map(|i| i as f64 * 0.01).collect();
//!
//! let result = solve(categories.view(), &y, None, &SolverParams::default(), None)
//!     .expect("solve should succeed");
//! assert!(result.converged);
//! ```
//!
//! # Module structure
//!
//! ```text
//! within
//! ├── observation         Storage backends (FactorMajorStore, ArrayStore)
//! ├── domain              Design<S> + factor-pair subdomains
//! │   └── factor_pairs      Domain construction, partition-of-unity weights
//! ├── operator            Linear algebra layer
//! │   ├── gramian           G = D^T W D (explicit CSR / implicit matvec)
//! │   │   ├── cross_tab       Bipartite block for a single factor pair
//! │   │   └── explicit        Full Gramian CSR assembly
//! │   ├── schwarz           Schwarz preconditioner builders (FE → generic API)
//! │   ├── preconditioner    FePreconditioner enum dispatch
//! │   ├── local_solver      ApproxChol + BlockElim backends
//! │   ├── schur_complement  Exact + approximate Schur (GKS 2023)
//! │   ├── residual_update   Observation-space vs sparse Gramian residuals
//! │   └── csr_block         Internal rectangular CSR off-diagonal block
//! ├── solver              Persistent Solver<S> with preconditioner reuse
//! ├── orchestrate         Public API: solve(), solve_batch()
//! ├── config              SolverParams, Preconditioner, LocalSolverConfig
//! └── error               WithinError, WithinResult<T>
//! ```
//!
//! # Crate dependency tree
//!
//! ```text
//! schwarz-precond           Generic domain decomposition (traits + Krylov solvers)
//! └── within                Fixed-effects solver (this crate)
//!     └── within-py         PyO3 bridge → python/within
//! ```
//!
//! # Where to start
//!
//! - **Using the API** — Start with the [`orchestrate`] module.
//!   [`solve()`](solve) and [`solve_batch()`](solve_batch) are the main entry
//!   points. For repeated solves with different right-hand sides, use
//!   [`Solver`] to amortize the preconditioner setup.
//!
//! - **Understanding the math** — Begin with
//!   [`operator::gramian`] to see how the Gramian is built and applied,
//!   then [`operator::schwarz`] for how factor-pair subdomains become a
//!   Schwarz preconditioner. The internal `operator::schur_complement`
//!   module implements block-elimination local solves.
//!
//! - **Extending with new backends** — The [`observation::Store`] trait
//!   abstracts over how factor-level data is laid out in memory. The
//!   [`schwarz_precond::LocalSolver`] trait (from the `schwarz-precond` crate)
//!   governs subdomain solvers.
//!
//! - **Tuning performance** — See the [`config`] module. [`SolverParams`]
//!   controls operator representation (implicit vs explicit Gramian),
//!   preconditioner variant (additive/multiplicative Schwarz, or none),
//!   Krylov method (CG/GMRES), tolerances, and local-solver settings.
//!
//! # Architecture
//!
//! The crate is organized in four layers:
//!
//! - **`observation`** — Per-observation data storage via [`FactorMajorStore`]
//!   and the [`observation::Store`] trait.
//! - **`domain`** — Domain decomposition: [`domain::Design`] wraps a store with
//!   factor metadata; factor-pair subdomains are built with partition-of-unity
//!   weights.
//! - **`operator`** — Linear algebra primitives: [`operator::gramian::Gramian`]
//!   (explicit CSR), the implicit [`operator::gramian::GramianOperator`], the
//!   rectangular [`operator::DesignOperator`] (both accept optional weights via
//!   their `new` constructor), and Schwarz preconditioner builders.
//! - **`orchestrate`** — End-to-end solve: [`solve`] with typed configuration.
//!
//! # References
//!
//! - Correia, S. (2017). *Linear Models with High-Dimensional Fixed Effects: An Efficient and Feasible Estimator.*
//! - Gao, Y., Kyng, R. & Spielman, D. A. (2025). AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization. *SIAM Journal on Scientific Computing*.
//! - Xu, J. (1992). Iterative Methods by Space Decomposition and Subspace Correction. *SIAM Review*, 34(4), 581--613.

pub mod config;
pub mod domain;
pub mod error;
pub mod observation;
pub mod operator;
pub mod orchestrate;
pub mod solver;
// ---------------------------------------------------------------------------
// Stable public API
// ---------------------------------------------------------------------------
//
// Lower-level types (Gramian, GramianOperator, DesignOperator, FeSchwarz,
// Design, Subdomain, ArrayStore, Store, LocalSolverConfig, ApproxCholConfig,
// ApproxSchurConfig, DEFAULT_DENSE_SCHUR_THRESHOLD) are intentionally not
// re-exported at the crate root. They remain accessible
// via their module paths (e.g. `within::operator::gramian::Gramian`,
// `within::config::LocalSolverConfig`) but are not part of the advertised
// stable surface.

pub use operator::preconditioner::FePreconditioner;
pub use orchestrate::{solve, solve_batch, BatchSolveResult, SolveResult};
pub use solver::Solver;

pub use config::{KrylovMethod, OperatorRepr, Preconditioner, ReductionStrategy, SolverParams};
pub use error::{WithinError, WithinResult};
pub use observation::FactorMajorStore;

// `Operator` is the trait `FePreconditioner` and the implicit operators
// implement; users need it in scope to call `apply` / `nrows` / `ncols`.
pub use schwarz_precond::Operator;
