//! Solver and preconditioner configuration types.
//!
//! `Option<&PreconditionerConfig>` accepts `None` (default Additive Schwarz),
//! `Some(Off)` (identity), `Some(Additive(_))` (tuned), or
//! `Some(Diagonal)` (Jacobi).

pub use schwarz_precond::ReductionStrategy;

/// Default `n_keep` threshold for dense Schur fast-path factorization.
///
/// Schur domains with `min(n_q, n_r) <= threshold` will first try dense
/// anchored Cholesky before falling back to sparse ApproxChol.
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

/// Local solver configuration for Schwarz subdomains.
///
/// Uses Schur complement reduction: eliminates the larger diagonal block
/// (exactly or approximately), then factorizes the smaller reduced system.
#[derive(Debug, Clone)]
pub struct LocalSolverConfig {
    /// ApproxChol config for the reduced system.
    pub approx_chol: ApproxCholConfig,
    /// Approximate Schur complement configuration.
    ///
    /// `Some(ApproxSchurConfig::default())` is the library default — approximate
    /// Schur with clique-tree sampling, which keeps per-subdomain factorization
    /// cost bounded under the iterative-solver context. `None` requests an
    /// exact Schur complement, used by tests and by callers who specifically
    /// want the higher-fidelity factorization.
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
            approx_chol: ApproxCholConfig {
                seed: 0,
                split_merge: Some(2),
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
