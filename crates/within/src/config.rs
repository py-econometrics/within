//! Solver and preconditioner configuration types.
//!
//! Configuration flows top-down through the crate's layers:
//!
//! ```text
//! SolverParams          (top-level: tolerance, max iterations, Krylov method)
//!   ├── OperatorRepr    (implicit D^T W D matvecs vs. explicit CSR Gramian)
//!   ├── KrylovMethod    (CG or GMRES with restart parameter)
//!   └── Preconditioner  (Schwarz variant + local solver config)
//!         ├── Additive(LocalSolverConfig, ReductionStrategy)
//!         └── Multiplicative(LocalSolverConfig)
//!               └── LocalSolverConfig { ApproxCholConfig, ApproxSchurConfig, dense_threshold }
//! ```
//!
//! # Defaults and why they are chosen
//!
//! | Parameter | Default | Rationale |
//! |---|---|---|
//! | `krylov` | CG | The Gramian G = D^T W D is symmetric positive semi-definite, so CG is optimal. Use GMRES only with non-symmetric (multiplicative Schwarz) preconditioners. |
//! | `operator` | Implicit | Avoids materializing the full Gramian in memory; only two cheap matvecs (D and D^T W) per CG iteration. Explicit is faster when the Gramian fits comfortably in cache. |
//! | `tol` | 1e-8 | Tight enough to preserve ~8 significant digits in the demeaned residuals, loose enough that well-preconditioned problems converge in tens of iterations. |
//! | `maxiter` | 1000 | Generous upper bound; well-preconditioned problems converge in tens of iterations. |
//! | `max_refinements` | 2 | Iterative refinement cheaply closes the gap between DOF-space and observation-space accuracy (see [`crate::solver`]). Rarely needs more than 1 actual correction. |
//! | `LocalSolverConfig` | SchurComplement | Schur reduction eliminates the larger diagonal block exactly, leaving a smaller system for approximate Cholesky. Much faster than factorizing the full SDDM system. |
//! | `dense_threshold` | 24 | Subdomains with `min(n_q, n_r) <= 24` use dense anchored Cholesky — exact and fast for small blocks. |
//!
//! # Usage from the public API
//!
//! Callers typically construct a [`SolverParams`] (possibly via `Default`) and
//! optionally a [`Preconditioner`], then pass both to [`crate::solve`] or
//! [`crate::Solver::new`]. The configuration is consumed during solver
//! construction and does not need to outlive the solver.

pub use schwarz_precond::ReductionStrategy;

/// Default `n_keep` threshold for dense Schur fast-path factorization.
///
/// Schur domains with `min(n_q, n_r) <= threshold` will first try dense
/// anchored Cholesky before falling back to sparse ApproxChol.
pub const DEFAULT_DENSE_SCHUR_THRESHOLD: usize = 24;

/// Configuration for approximate Cholesky factorization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApproxCholConfig {
    /// Random seed for the factorization sampler.
    pub seed: u64,
    /// Optional split/merge count for denser AC2-style factorizations.
    pub split_merge: Option<u32>,
}

impl ApproxCholConfig {
    pub(crate) fn to_approx_chol(self) -> approx_chol::Config {
        approx_chol::Config {
            seed: self.seed,
            split_merge: self.split_merge,
        }
    }
}

// ---------------------------------------------------------------------------
// Operator representation
// ---------------------------------------------------------------------------

/// Operator representation for normal equations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OperatorRepr {
    /// Matrix-free `D^T W D x` — no Gramian stored; recomputes each matvec.
    #[default]
    Implicit,
    /// Pre-assembled CSR Gramian — one-time build, then O(nnz) matvecs.
    Explicit,
}

// ---------------------------------------------------------------------------
// Local solver configuration
// ---------------------------------------------------------------------------

/// Local solver configuration for Schwarz subdomains.
///
/// Uses Schur complement reduction: eliminates the larger diagonal block
/// (exactly or approximately), then factorizes the smaller reduced system.
#[derive(Debug, Clone)]
pub struct LocalSolverConfig {
    /// ApproxChol config for the reduced system.
    pub approx_chol: ApproxCholConfig,
    /// Approximate Schur complement configuration.
    /// `None` = exact (default). `Some` = approximate with sampling.
    pub approx_schur: Option<ApproxSchurConfig>,
    /// Dense Schur fast-path threshold on reduced size `n_keep=min(n_q,n_r)`.
    ///
    /// `0` disables the dense fast path; larger values allow dense anchored
    /// Cholesky for more subdomains.
    pub dense_threshold: usize,
}

impl Default for LocalSolverConfig {
    fn default() -> Self {
        Self {
            approx_chol: ApproxCholConfig::default(),
            approx_schur: Some(ApproxSchurConfig::default()),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        }
    }
}

impl LocalSolverConfig {
    /// Default for iterative solvers: uses split_merge=2 for the reduced Schur system.
    pub fn solver_default() -> Self {
        Self {
            approx_chol: ApproxCholConfig {
                split_merge: Some(2),
                ..Default::default()
            },
            approx_schur: Some(ApproxSchurConfig::default()),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        }
    }
}

// ---------------------------------------------------------------------------
// Approximate Schur complement configuration
// ---------------------------------------------------------------------------

/// Configuration for approximate Schur complement via clique-tree sampling.
///
/// Every eliminated vertex uses a sampled spanning tree (at most deg-1 fill
/// edges) via the GKS 2023 Algorithm 3 clique-tree. This preserves spectral
/// quality (unbiased edge weights) while reducing fill-in to O(deg).
///
/// When `split > 1`, each edge in the star is split into `split` parallel
/// copies (each carrying `1/split` of the original weight) before sampling
/// the clique-tree. This produces up to `split * (deg-1)` fill edges,
/// giving a denser (better) Schur approximation at the cost of more fill-in.
#[derive(Debug, Clone, Copy)]
pub struct ApproxSchurConfig {
    /// Random seed for the clique-tree sampler.
    pub seed: u64,
    /// Edge split factor: each star edge is split into `split` copies
    /// before clique-tree sampling.
    ///
    /// `1` = no splitting (standard), `k > 1` = denser approximation.
    pub split: u32,
}

impl Default for ApproxSchurConfig {
    fn default() -> Self {
        Self { seed: 0, split: 1 }
    }
}

// ---------------------------------------------------------------------------
// Krylov method
// ---------------------------------------------------------------------------

/// Outer Krylov method.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KrylovMethod {
    /// Preconditioned conjugate gradient (requires symmetric preconditioner).
    #[default]
    Cg,
    /// Right-preconditioned GMRES with restart.
    Gmres {
        /// Restart dimension (number of Arnoldi vectors before restart).
        restart: usize,
    },
    /// Modified LSMR: rectangular least-squares, minimum normal-equation residual.
    ///
    /// Uses Modified Golub-Kahan bidiagonalization — one `M⁻¹` solve per iteration.
    /// The preconditioner `M ≈ A^T A` is used directly without square-root factorization.
    Lsmr {
        /// Number of past `v` vectors to reorthogonalize against via
        /// windowed modified Gram-Schmidt. `None` (default) disables — the
        /// plain short recurrence is used. `Some(N)` enables a window of
        /// `N` past vectors; `Some(5..20)` is cheap insurance for
        /// ill-conditioned problems where rounding causes the
        /// bidiagonalization to lose orthogonality and convergence to
        /// stall. Memory cost is `local_size · n` doubles unpreconditioned,
        /// `2·local_size · n` preconditioned.
        local_size: Option<usize>,
    },
}

// ---------------------------------------------------------------------------
// Preconditioner
// ---------------------------------------------------------------------------

/// Schwarz preconditioner variant with embedded local solver configuration.
///
/// CG requires a symmetric preconditioner so only `Additive` is valid.
/// GMRES supports both.
#[derive(Debug, Clone)]
pub enum Preconditioner {
    /// Additive Schwarz (symmetric — valid for CG and GMRES).
    Additive(LocalSolverConfig, ReductionStrategy),
    /// Multiplicative Schwarz (non-symmetric — GMRES only).
    Multiplicative(LocalSolverConfig),
}

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

/// Top-level solver configuration: Krylov method, operator representation, and tolerances.
#[derive(Debug, Clone)]
pub struct SolverParams {
    /// Krylov subspace method (CG or GMRES).
    pub krylov: KrylovMethod,
    /// Operator representation for the normal equations.
    pub operator: OperatorRepr,
    /// Relative residual convergence tolerance.
    pub tol: f64,
    /// Maximum Krylov iterations before declaring non-convergence.
    pub maxiter: usize,
    /// Maximum number of iterative refinement steps after the initial Krylov solve.
    ///
    /// Iterative refinement recomputes the normal-equation residual from observation
    /// space (`D^T W (y - D x)`) and solves for a correction. This closes the gap
    /// between normal-equation residual accuracy and observation-space demeaning
    /// quality that arises when the Gramian condition number is large.
    ///
    /// Each refinement step costs two cheap matvecs (D and D^T W) plus one Krylov
    /// solve with a small RHS. The check alone (without the solve) is nearly free.
    /// Typically 0–1 actual correction solves are needed.
    pub max_refinements: usize,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            krylov: KrylovMethod::Cg,
            operator: OperatorRepr::Implicit,
            tol: 1e-8,
            maxiter: 1000,
            max_refinements: 2,
        }
    }
}
