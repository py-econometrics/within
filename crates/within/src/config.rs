//! Solver and preconditioner configuration types.
//!
//! `Option<&PreconditionerConfig>` accepts `None` (default Additive Schwarz),
//! `Some(Off)` (identity), `Some(Additive(_))` (tuned), or
//! `Some(Diagonal)` (Jacobi).
//!
//! Stability policy: enums that may gain variants (the preconditioner strategy
//! set) stay `#[non_exhaustive]`, so adding a variant is non-breaking — external
//! `match` sites already carry a wildcard arm. Option structs commit to public
//! fields and literal construction; adding a field is a deliberate breaking
//! change, caught mechanically by the cargo-semver-checks CI gate rather than
//! slipping through silently.

pub use schwarz_precond::ReductionStrategy;

/// Default `n_keep` threshold below which a Schur domain tries the exact dense backend before falling back to sampled.
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
    pub(crate) fn to_approx_chol(
        self,
        exact_below: usize,
        on_failure: approx_chol::ExactFailure,
    ) -> approx_chol::Config {
        approx_chol::Config {
            seed: self.seed,
            split_merge: self.split_merge,
            // `max_dim: 0` claims no block, which is what `dense_threshold: 0` asks for.
            backend: approx_chol::Backend::ExactBelow {
                max_dim: exact_below,
                on_failure,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Local solver configuration
// ---------------------------------------------------------------------------

/// Schur-complement reduction mode: sampled keeps per-subdomain factorization cost bounded, exact trades speed for fidelity.
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
#[derive(Debug, Clone)]
pub struct LocalSolverConfig {
    /// ApproxChol config for the reduced system.
    pub approx_chol: ApproxCholConfig,
    /// Schur-complement reduction mode (default: approximate).
    pub schur: SchurMode,
    /// Exact-Schur threshold on reduced size `min(n_rows, n_cols)`, at or below which fill-in is affordable. `0` disables.
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

/// Certification policy for the diagonal scaling to SDDM form. Frustration is always a hard error; this governs only the dominance certificate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingConfig {
    /// Relative slack for weak diagonal dominance; PSD-boundary designs hover at ≈ 1e-12, so the default does not flip on rounding luck.
    pub tolerance: f64,
    /// Sweep budget for the dominance relaxation.
    pub max_sweeps: usize,
    /// Disposition when certification fails.
    pub on_failure: ScalingFailure,
}

/// What to do when a component's dominance scaling cannot be certified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingFailure {
    /// Clamp residual deficits — preconditioner quality only — and record a [`BuildWarning`](crate::BuildWarning).
    Warn,
    /// Fail the build with [`BuildError::UnscalableComponent`](crate::BuildError::UnscalableComponent).
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

/// Approximate Schur via GKS 2023 Algorithm 3 clique-tree sampling: unbiased edge weights at O(deg) fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApproxSchurConfig {
    /// Random seed for the clique-tree sampler.
    pub seed: u64,
    /// Star-edge split factor: `1` is standard, `k > 1` a denser approximation at more fill-in.
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

/// Preconditioner variant. `#[non_exhaustive]`, so external `match` sites need a wildcard arm.
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
    /// Diagonal/Jacobi `M^{-1} = diag(DᵀWD)^{-1}`; a zero diagonal takes the pseudo-inverse, pinning that coordinate to 0.
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
    /// Window of past `v` vectors for windowed modified Gram-Schmidt; `None` disables. Costs `local_size · n` doubles, `2×` preconditioned.
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
