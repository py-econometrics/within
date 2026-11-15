//! Solver and preconditioner configuration types.
//!
//! Stability policy: option structs commit to public fields and literal construction; adding
//! a field is a deliberate breaking change, caught mechanically by the cargo-semver-checks CI
//! gate rather than slipping through silently.

use serde::{Deserialize, Serialize};

use crate::error::BuildError;

pub use schwarz_precond::{Staleness, StalenessError};

/// Default block dimension at or below which a block is factored exactly.
pub(crate) const DEFAULT_DENSE_SCHUR_THRESHOLD: usize = 24;

/// Configuration for approximate Cholesky factorization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApproxCholConfig {
    /// Random seed for the factorization sampler.
    pub seed: u64,
    /// Optional split/merge count for denser AC2-style factorizations.
    pub split_merge: Option<u32>,
}

/// Schur reduction mode: sampled bounds per-subdomain cost, exact trades speed for fidelity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalSolverConfig {
    /// ApproxChol config for the reduced system.
    pub approx_chol: ApproxCholConfig,
    /// Schur-complement reduction mode (default: approximate).
    pub schur: SchurMode,
    /// Largest Cholesky dimension factored exactly, a signed block's two cover sheets counted; `0` disables.
    pub dense_threshold: usize,
    /// Certification policy for the diagonal scaling of signed components.
    pub scaling: ScalingConfig,
}

impl LocalSolverConfig {
    /// Separate from any build so a deferred escalation can reject the config at construction.
    pub(crate) fn validate(&self) -> Result<(), BuildError> {
        self.scaling_policy().map(|_| ())
    }

    pub(crate) fn scaling_policy(&self) -> Result<approx_chol::ScalingPolicy, BuildError> {
        let ScalingConfig {
            tolerance,
            max_iterations,
            ..
        } = self.scaling;
        approx_chol::ScalingPolicy::new(tolerance, max_iterations)
            .ok_or(BuildError::InvalidScalingTolerance { value: tolerance })
    }

    pub(crate) fn factor_config(&self) -> approx_chol::Config {
        // `split_merge: None` and a split of `1` are standard AC; a count above one is AC2.
        let sampling = |copies: Option<u32>| approx_chol::Sampling::CliqueTree {
            copies: copies
                .and_then(std::num::NonZeroU32::new)
                .unwrap_or(std::num::NonZeroU32::MIN),
        };
        approx_chol::Config {
            seed: self.approx_chol.seed,
            sampling: sampling(self.approx_chol.split_merge),
            exact_max_dim: self.dense_threshold,
            complement: match self.schur {
                SchurMode::Approximate(ApproxSchurConfig { split }) => {
                    approx_chol::SchurMode::Approximate {
                        sampling: sampling(Some(split)),
                    }
                }
                SchurMode::Exact => approx_chol::SchurMode::Exact,
            },
            ..Default::default()
        }
    }
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

/// Frustration is always a hard error; this governs only the dominance certificate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Relative slack for weak dominance, in `[0, 1)`; PSD-boundary designs hover at ≈ 1e-12.
    pub tolerance: f64,
    /// Scaling iteration budget per component.
    pub max_iterations: usize,
    /// Disposition when certification fails.
    pub on_failure: ScalingFailure,
}

/// What to do when a component's dominance scaling cannot be certified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingFailure {
    /// Clamp residual deficits and record a [`BuildWarning`](crate::BuildWarning).
    Warn,
    /// Fail with [`BuildError::UnscalableComponent`].
    Error,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-9,
            max_iterations: 2048,
            on_failure: ScalingFailure::Warn,
        }
    }
}

/// Approximate Schur via GKS 2023 Algorithm 3, drawing from [`ApproxCholConfig::seed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApproxSchurConfig {
    /// Star-edge split factor: `1` is standard, `k > 1` a denser approximation at more fill-in.
    pub split: u32,
}

impl Default for ApproxSchurConfig {
    fn default() -> Self {
        Self { split: 1 }
    }
}

/// The preconditioner a solve runs on, and the one it moves to when that one stalls.
// Append only: postcard encodes by declaration order and every `Preconditioner` carries one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PreconditionerConfig {
    /// Identity preconditioner. Solves the unpreconditioned normal equations.
    Off,
    /// Diagonal/Jacobi; a zero diagonal takes the pseudo-inverse, pinning that coordinate to 0.
    Diagonal,
    /// Schwarz from construction on.
    Schwarz(Schwarz),
    /// Diagonal first; Schwarz is built at, and used from, the first `stall`.
    Adaptive {
        /// The contraction stall on the diagonal that hands the solve to Schwarz.
        stall: Staleness,
        /// The Schwarz the solve moves to.
        schwarz: Schwarz,
    },
}

impl Default for PreconditionerConfig {
    fn default() -> Self {
        Self::Adaptive {
            stall: Staleness::default(),
            schwarz: Schwarz::default(),
        }
    }
}

impl PreconditionerConfig {
    /// Its rungs in the order a solve reaches them.
    pub(crate) fn rungs(&self) -> impl Iterator<Item = RungSettings<'_>> {
        let (diagonal, schwarz) = match self {
            Self::Off => (None, None),
            Self::Diagonal => (Some(RungSettings::Diagonal { stall: None }), None),
            Self::Schwarz(schwarz) => (None, Some(schwarz)),
            Self::Adaptive { stall, schwarz } => (
                Some(RungSettings::Diagonal { stall: Some(stall) }),
                Some(schwarz),
            ),
        };
        diagonal
            .into_iter()
            .chain(schwarz.map(RungSettings::Schwarz))
    }

    /// Where `rung` stands among its rungs; `None` when it is not one of them.
    pub(crate) fn position(&self, rung: Rung) -> Option<usize> {
        self.rungs().position(|r| r.rung() == rung)
    }

    pub(crate) fn validate(&self) -> Result<(), BuildError> {
        match self {
            Self::Off | Self::Diagonal => Ok(()),
            Self::Schwarz(schwarz) | Self::Adaptive { schwarz, .. } => {
                schwarz.local_solver.validate()
            }
        }
    }
}

/// One-level additive Schwarz over factor-pair subdomains.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Schwarz {
    /// Local solver configuration applied inside each subdomain.
    pub local_solver: LocalSolverConfig,
}

/// A preconditioner a solve can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rung {
    /// Diagonal/Jacobi.
    Diagonal,
    /// One-level additive Schwarz.
    Schwarz,
}

/// One rung's settings, as [`PreconditionerConfig::rungs`] yields them.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RungSettings<'c> {
    /// `stall` hands the solve to Schwarz.
    Diagonal {
        stall: Option<&'c Staleness>,
    },
    Schwarz(&'c Schwarz),
}

impl<'c> RungSettings<'c> {
    pub(crate) fn rung(&self) -> Rung {
        match self {
            Self::Diagonal { .. } => Rung::Diagonal,
            Self::Schwarz(_) => Rung::Schwarz,
        }
    }

    /// The stall on this rung that hands the solve on; `None` on a rung that is never left.
    pub(crate) fn stall(&self) -> Option<&'c Staleness> {
        match self {
            Self::Diagonal { stall } => *stall,
            Self::Schwarz(_) => None,
        }
    }
}

/// LSMR solver configuration: tolerances and reorthogonalization window.
#[derive(Debug, Clone)]
pub struct LsmrOptions {
    /// Relative residual convergence tolerance.
    pub tol: f64,
    /// Maximum LSMR iterations before declaring non-convergence.
    pub maxiter: usize,
    /// Window of past `v` vectors for windowed MGS; `None` disables. Costs `local_size · n`.
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
