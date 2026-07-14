//! Solver and preconditioner configuration types.
//!
//! `Option<&PreconditionerConfig>` accepts `None` (default Additive Schwarz),
//! `Some(Off)` (identity), `Some(Additive(_))` (tuned), or
//! `Some(Diagonal)` (Jacobi).

pub use schwarz_precond::ReductionStrategy;

/// Default `n_keep` threshold for dense Schur fast-path factorization.
///
/// Schur domains with `min(n_q, n_r) <= threshold` first try dense Cholesky
/// before falling back to sparse ApproxChol.
pub(crate) const DEFAULT_DENSE_SCHUR_THRESHOLD: usize = 24;

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
// Local solver configuration
// ---------------------------------------------------------------------------

/// Schur-complement reduction mode for the local solver.
///
/// [`Approximate`](Self::Approximate) (the default) uses clique-tree sampling,
/// which keeps per-subdomain factorization cost bounded under the iterative
/// solver. [`Exact`](Self::Exact) forms the exact Schur complement — higher
/// fidelity, slower per subdomain — for validation and callers who want it.
#[derive(Debug, Clone, PartialEq)]
pub enum SchurMode {
    /// Approximate Schur via clique-tree sampling.
    Approximate(ApproxSchurConfig),
    /// Exact Schur complement.
    Exact,
}

impl Default for SchurMode {
    fn default() -> Self {
        SchurMode::Approximate(ApproxSchurConfig::default())
    }
}

/// Local solver configuration for Schwarz subdomains.
///
/// Uses Schur complement reduction: eliminates the larger diagonal block
/// (exactly or approximately), then factorizes the smaller reduced system.
#[derive(Debug, Clone)]
pub struct LocalSolverConfig {
    /// ApproxChol config for the reduced system.
    pub approx_chol: ApproxCholConfig,
    /// Schur-complement reduction mode (default: approximate).
    pub schur: SchurMode,
    /// Dense Schur fast-path threshold on reduced size `n_keep=min(n_q,n_r)`.
    ///
    /// `0` disables the dense fast path; larger values allow dense Cholesky for
    /// more subdomains.
    pub dense_threshold: usize,
    /// Certification policy for the diagonal scaling of signed components.
    pub scaling: ScalingConfig,
}

impl Default for LocalSolverConfig {
    fn default() -> Self {
        Self {
            approx_chol: ApproxCholConfig {
                seed: 0,
                split_merge: Some(2),
            },
            schur: SchurMode::default(),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
            scaling: ScalingConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Signed-component scaling configuration
// ---------------------------------------------------------------------------

/// Certification policy for the diagonal scaling that converts signed
/// components to SDDM form.
///
/// Frustration (a negative-sign cycle) is always a hard build error; this
/// governs only the diagonal-dominance certification of the scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingConfig {
    /// Relative slack for accepting weak diagonal dominance.
    ///
    /// Real PSD-boundary designs (unit trends, cohort+time slope spans) hover
    /// at violations ≈ 1e-12; the default keeps them comfortably inside the
    /// accepted band instead of flipping on rounding luck.
    pub tolerance: f64,
    /// Sweep budget for the dominance relaxation.
    pub max_sweeps: usize,
    /// Disposition when certification fails.
    pub on_failure: ScalingFailure,
}

/// What to do when a component's dominance scaling cannot be certified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingFailure {
    /// Clamp residual deficits — preconditioner quality only — and record a
    /// [`BuildWarning`](crate::BuildWarning).
    Warn,
    /// Fail the build with
    /// [`BuildError::UnscalableComponent`](crate::BuildError::UnscalableComponent).
    Error,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-9,
            max_sweeps: 2048,
            on_failure: ScalingFailure::Warn,
        }
    }
}

// ---------------------------------------------------------------------------
// Approximate Schur complement configuration
// ---------------------------------------------------------------------------

/// Configuration for approximate Schur complement via clique-tree sampling.
///
/// Every eliminated vertex uses a sampled spanning tree via the GKS 2023
/// Algorithm 3 clique-tree. In grounded systems, ground is an ordinary member
/// of the star. This preserves unbiased edge weights with O(deg) fill.
///
/// When `split > 1`, each edge in the star is split into `split` parallel
/// copies (each carrying `1/split` of the original weight) before sampling
/// the clique-tree. This produces a denser Schur approximation at the cost of
/// more fill-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
// Preconditioner configuration
// ---------------------------------------------------------------------------

/// Preconditioner variant.
///
/// `#[non_exhaustive]` — external `match` sites must include a wildcard arm.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PreconditionerConfig {
    /// Identity preconditioner. Solves the unpreconditioned normal equations.
    Off,
    /// One-level additive Schwarz over factor-pair subdomains.
    Additive {
        /// Local solver configuration applied inside each subdomain.
        local_solver: LocalSolverConfig,
        /// Strategy for combining overlapping subdomain contributions.
        reduction: ReductionStrategy,
    },
    /// Diagonal/Jacobi preconditioner with `M^{-1} = diag(D^T W D)^{-1}`.
    ///
    /// A level with no observations (or one that is fully zero-weighted) has a
    /// zero diagonal; it takes the pseudo-inverse (`inv = 0`), pinning that
    /// coordinate to 0 as on the unpreconditioned path.
    Diagonal,
}

impl Default for PreconditionerConfig {
    fn default() -> Self {
        Self::Additive {
            local_solver: LocalSolverConfig::default(),
            reduction: ReductionStrategy::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// LSMR configuration
// ---------------------------------------------------------------------------

/// LSMR solver configuration: tolerances and reorthogonalization window.
#[derive(Debug, Clone)]
pub struct LsmrOptions {
    /// Relative residual convergence tolerance.
    pub tol: f64,
    /// Maximum LSMR iterations before declaring non-convergence.
    pub maxiter: usize,
    /// Number of past `v` vectors to reorthogonalize against via windowed
    /// modified Gram-Schmidt. `None` (default) disables — the plain short
    /// recurrence is used. `Some(N)` enables a window of `N` past vectors;
    /// `Some(5..20)` is cheap insurance for ill-conditioned problems where
    /// rounding causes the bidiagonalization to lose orthogonality and
    /// convergence to stall. Memory cost is `local_size · n` doubles
    /// unpreconditioned, `2·local_size · n` preconditioned.
    pub local_size: Option<usize>,
}

impl Default for LsmrOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            maxiter: 1000,
            local_size: None,
        }
    }
}
