//! Solver and preconditioner configuration types.

/// Default `n_keep` threshold for dense Schur fast-path factorization.
///
/// Schur domains with `min(n_q, n_r) <= threshold` will first try dense
/// anchored Cholesky before falling back to sparse ApproxChol.
pub const DEFAULT_DENSE_SCHUR_THRESHOLD: usize = 24;

// ---------------------------------------------------------------------------
// Solver configuration types
// ---------------------------------------------------------------------------

/// Selects the local solver used inside each Schwarz subdomain.
#[derive(Debug, Clone)]
pub enum LocalSolverConfig {
    /// Full bipartite SDDM factorized via approximate Cholesky.
    ///
    /// Uses `approx_chol` from the parent `SchwarzConfig`.
    FullSddm,
    /// Schur complement reduction: eliminate the larger diagonal block
    /// exactly, then factorize the smaller reduced system.
    SchurComplement {
        /// ApproxChol config for the reduced system. Defaults to AC2(2,2).
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
            approx_chol: approx_chol::Config {
                split_merge: Some(8),
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
/// edges) via the GKS 2023 Algorithm 5 clique-tree. This preserves spectral
/// quality (unbiased edge weights) while reducing fill-in to O(deg).
#[derive(Debug, Clone, Copy, Default)]
pub struct ApproxSchurConfig {
    /// Random seed for the clique-tree sampler.
    pub seed: u64,
}

/// Common configuration for all Schwarz-based preconditioners.
#[derive(Debug, Clone, Default)]
pub struct SchwarzConfig {
    pub approx_chol: approx_chol::Config,
    pub local_solver: LocalSolverConfig,
}

/// CG preconditioner configuration.
#[derive(Debug, Clone, Default)]
pub enum CgPreconditioner {
    /// Unpreconditioned CG.
    #[default]
    None,
    /// One-level additive Schwarz.
    OneLevel(SchwarzConfig),
    /// One-level multiplicative Schwarz (symmetric: forward + backward).
    MultiplicativeOneLevel(SchwarzConfig),
}

/// GMRES preconditioner configuration (forward-only multiplicative Schwarz).
#[derive(Debug, Clone)]
pub enum GmresPreconditioner {
    /// One-level multiplicative Schwarz (forward-only).
    MultiplicativeOneLevel(SchwarzConfig),
}

#[derive(Debug, Clone)]
pub enum SolverMethod {
    Lsmr {
        conlim: f64,
    },
    Cg {
        preconditioner: CgPreconditioner,
    },
    Gmres {
        preconditioner: GmresPreconditioner,
        restart: usize,
    },
}

impl Default for SolverMethod {
    fn default() -> Self {
        Self::Lsmr { conlim: 1e8 }
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
