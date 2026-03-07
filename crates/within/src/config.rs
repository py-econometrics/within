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
    /// Default for iterative solvers: uses split_merge=2 for the reduced Schur system.
    pub fn solver_default() -> Self {
        Self::SchurComplement {
            approx_chol: approx_chol::Config {
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
/// edges) via the GKS 2023 Algorithm 5 clique-tree. This preserves spectral
/// quality (unbiased edge weights) while reducing fill-in to O(deg).
#[derive(Debug, Clone, Copy, Default)]
pub struct ApproxSchurConfig {
    /// Random seed for the clique-tree sampler.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// Krylov method
// ---------------------------------------------------------------------------

/// Outer Krylov method.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KrylovMethod {
    #[default]
    Cg,
    Gmres {
        restart: usize,
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
    Additive(LocalSolverConfig),
    /// Multiplicative Schwarz (non-symmetric — GMRES only).
    Multiplicative(LocalSolverConfig),
}

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SolverParams {
    pub krylov: KrylovMethod,
    pub operator: OperatorRepr,
    pub preconditioner: Option<Preconditioner>,
    pub tol: f64,
    pub maxiter: usize,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            krylov: KrylovMethod::Cg,
            operator: OperatorRepr::Implicit,
            preconditioner: Some(Preconditioner::Additive(LocalSolverConfig::solver_default())),
            tol: 1e-8,
            maxiter: 1000,
        }
    }
}
