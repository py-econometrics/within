//! Solver and preconditioner configuration types.

/// Default `n_keep` threshold for dense Schur fast-path factorization.
///
/// Schur domains with `min(n_q, n_r) <= threshold` will first try dense
/// anchored Cholesky before falling back to sparse ApproxChol.
pub const DEFAULT_DENSE_SCHUR_THRESHOLD: usize = 24;

// ---------------------------------------------------------------------------
// Operator representation
// ---------------------------------------------------------------------------

/// Operator representation for normal equations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OperatorRepr {
    #[default]
    Implicit,
    Explicit,
}

// ---------------------------------------------------------------------------
// Local solver configuration
// ---------------------------------------------------------------------------

/// Selects the local solver used inside each Schwarz subdomain.
///
/// Each variant is self-contained: it carries its own `approx_chol::Config`.
#[derive(Debug, Clone)]
pub enum LocalSolverConfig {
    /// Full bipartite SDDM factorized via approximate Cholesky.
    FullSddm { approx_chol: approx_chol::Config },
    /// Schur complement reduction: eliminate the larger diagonal block
    /// exactly, then factorize the smaller reduced system.
    SchurComplement {
        /// ApproxChol config for the reduced system.
        approx_chol: approx_chol::Config,
        /// Approximate Schur complement configuration.
        /// `None` = exact (default). `Some` = approximate with sampling.
        approx_schur: Option<ApproxSchurConfig>,
        /// Dense Schur fast-path threshold on reduced size `n_keep=min(n_q,n_r)`.
        ///
        /// `0` disables the dense fast path; larger values allow dense anchored
        /// Cholesky for more subdomains.
        dense_threshold: usize,
    },
}

impl Default for LocalSolverConfig {
    fn default() -> Self {
        Self::SchurComplement {
            approx_chol: approx_chol::Config::default(),
            approx_schur: Some(ApproxSchurConfig::default()),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        }
    }
}

impl LocalSolverConfig {
    /// CG-specific default: uses split_merge=2 for the reduced Schur system.
    pub fn cg_default() -> Self {
        Self::SchurComplement {
            approx_chol: approx_chol::Config {
                split_merge: Some(2),
                ..Default::default()
            },
            approx_schur: Some(ApproxSchurConfig::default()),
            dense_threshold: DEFAULT_DENSE_SCHUR_THRESHOLD,
        }
    }

    /// GMRES-specific default: uses split_merge=2 for the reduced Schur system.
    pub fn gmres_default() -> Self {
        Self::cg_default()
    }
}

// ---------------------------------------------------------------------------
// Approximate Schur complement configuration
// ---------------------------------------------------------------------------

/// Configuration for approximate Schur complement via clique-tree sampling.
///
/// Every eliminated vertex uses a sampled spanning tree (at most deg-1 fill
/// edges) via the GKS 2023 Algorithm 5 clique-tree. This preserves spectral
/// quality (unbiased edge weights) while reducing fill-in to O(deg).
#[derive(Debug, Clone, Copy, Default)]
pub struct ApproxSchurConfig {
    /// Random seed for the clique-tree sampler.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// GMRES preconditioner choice
// ---------------------------------------------------------------------------

/// Preconditioner choice for GMRES (supports both additive and multiplicative).
#[derive(Debug, Clone)]
pub enum GmresPrecond {
    Additive(LocalSolverConfig),
    Multiplicative(LocalSolverConfig),
}

// ---------------------------------------------------------------------------
// Solver configuration types
// ---------------------------------------------------------------------------

/// Solver method with embedded preconditioner choice.
///
/// CG can only use additive Schwarz (or none) — multiplicative is impossible
/// at the type level since it produces a non-symmetric preconditioner.
#[derive(Debug, Clone)]
pub enum SolverMethod {
    Cg {
        /// `None` = unpreconditioned, `Some` = additive Schwarz.
        preconditioner: Option<LocalSolverConfig>,
        operator: OperatorRepr,
    },
    Gmres {
        /// `None` = unpreconditioned, `Some` = additive or multiplicative.
        preconditioner: Option<GmresPrecond>,
        operator: OperatorRepr,
        restart: usize,
    },
}

impl Default for SolverMethod {
    fn default() -> Self {
        Self::cg_default()
    }
}

impl SolverMethod {
    /// CG with additive Schwarz, implicit operator.
    pub fn cg_default() -> Self {
        Self::Cg {
            preconditioner: Some(LocalSolverConfig::cg_default()),
            operator: OperatorRepr::Implicit,
        }
    }

    /// GMRES with additive Schwarz, implicit operator, restart=30.
    pub fn gmres_default() -> Self {
        Self::Gmres {
            preconditioner: Some(GmresPrecond::Additive(LocalSolverConfig::gmres_default())),
            operator: OperatorRepr::Implicit,
            restart: 30,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolverParams {
    pub method: SolverMethod,
    pub tol: f64,
    pub maxiter: usize,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            method: SolverMethod::default(),
            tol: 1e-8,
            maxiter: 1000,
        }
    }
}
