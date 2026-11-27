//! Solver and preconditioner configuration types.
//!
//! Stability policy: enums that may gain variants (the preconditioner strategy
//! set) stay `#[non_exhaustive]`, so adding a variant is non-breaking — external
//! `match` sites already carry a wildcard arm. Option structs commit to public
//! fields and literal construction; adding a field is a deliberate breaking
//! change, caught mechanically by the cargo-semver-checks CI gate rather than
//! slipping through silently.

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

/// Preconditioner variant. `#[non_exhaustive]`, so external `match` sites need a wildcard arm.
// Append only: postcard encodes by declaration order and every `Preconditioner` carries one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PreconditionerConfig {
    /// Identity preconditioner. Solves the unpreconditioned normal equations.
    Off,
    /// Schwarz from construction on.
    Schwarz(Schwarz),
    /// Diagonal/Jacobi; a zero diagonal takes the pseudo-inverse, pinning that coordinate to 0.
    Diagonal,
    /// Diagonal first; Schwarz is built at, and used from, the first `stall`.
    Adaptive {
        /// The contraction stall on the diagonal that hands the solve to Schwarz.
        stall: Staleness,
        /// The Schwarz the solve moves to.
        schwarz: Schwarz,
    },
}

/// One-level additive Schwarz over channel-pair subdomains.
// Append only: postcard encodes fields in declaration order and every `Preconditioner` carries one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Schwarz {
    /// Local solver configuration applied inside each subdomain.
    pub local_solver: LocalSolverConfig,
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
    /// The rungs a solve climbs, in order, each with the stall that hands a solve to the next.
    ///
    /// PROTOTYPE(#607): `PROBE_COARSE=climb` adds a coarse rung above Adaptive's Schwarz,
    /// `PROBE_COARSE=fixed` makes the Schwarz config build the coarse correction at once;
    /// `PROBE_STALL1` / `PROBE_STALL2` override the first and second climb's stall rule.
    pub(crate) fn rungs(&self) -> impl Iterator<Item = (Rung<'_>, Option<&Staleness>)> {
        let probe = probe::knobs();
        let rungs: [Option<(Rung<'_>, Option<&Staleness>)>; 3] = match self {
            Self::Off => [None, None, None],
            Self::Diagonal => [Some((Rung::Diagonal, None)), None, None],
            Self::Schwarz(schwarz) => match probe.coarse {
                probe::Coarse::Fixed => [Some((Rung::Coarse(schwarz), None)), None, None],
                _ => [Some((Rung::Schwarz(schwarz), None)), None, None],
            },
            Self::Adaptive { stall, schwarz } => {
                let first = probe.stall1.as_ref().unwrap_or(stall);
                match probe.coarse {
                    probe::Coarse::Climb => [
                        Some((Rung::Diagonal, Some(first))),
                        Some((
                            Rung::Schwarz(schwarz),
                            Some(probe.stall2.as_ref().unwrap_or(first)),
                        )),
                        Some((Rung::Coarse(schwarz), None)),
                    ],
                    _ => [
                        Some((Rung::Diagonal, Some(first))),
                        Some((Rung::Schwarz(schwarz), None)),
                        None,
                    ],
                }
            }
        };
        rungs.into_iter().flatten()
    }

    pub(crate) fn validate(&self) -> Result<(), BuildError> {
        self.rungs().try_for_each(|(rung, _)| rung.validate())
    }
}

/// The map one rung of a ladder builds.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Rung<'c> {
    Diagonal,
    Schwarz(&'c Schwarz),
    /// PROTOTYPE(#607): Schwarz plus the rigid coarse correction; builds Schwarz itself unless
    /// the rung below holds it.
    Coarse(&'c Schwarz),
}

impl Rung<'_> {
    /// Separate from any build so a deferred rung can reject its config at construction.
    fn validate(&self) -> Result<(), BuildError> {
        match self {
            Self::Diagonal => Ok(()),
            Self::Schwarz(schwarz) | Self::Coarse(schwarz) => schwarz.local_solver.validate(),
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

/// PROTOTYPE(#607): the probe's knobs, read once from the environment.
pub(crate) mod probe {
    use std::sync::OnceLock;

    use super::Staleness;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Coarse {
        Off,
        Climb,
        Fixed,
    }

    /// The bottom rung: today's diagonal, or block Jacobi on `S`'s level blocks.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Bottom {
        Diagonal,
        Block,
    }

    #[derive(Debug)]
    pub(crate) struct Knobs {
        pub(crate) coarse: Coarse,
        pub(crate) bottom: Bottom,
        pub(crate) stall1: Option<Staleness>,
        pub(crate) stall2: Option<Staleness>,
    }

    fn stall(name: &str) -> Option<Staleness> {
        let raw = std::env::var(name).ok()?;
        let (w, t) = raw
            .split_once(',')
            .unwrap_or_else(|| panic!("{name}={raw}: want w,t"));
        let window = w
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name}={raw}: bad window"));
        let threshold = t
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name}={raw}: bad threshold"));
        Some(Staleness::try_new(window, threshold).unwrap_or_else(|e| panic!("{name}={raw}: {e}")))
    }

    pub(crate) fn knobs() -> &'static Knobs {
        static KNOBS: OnceLock<Knobs> = OnceLock::new();
        KNOBS.get_or_init(|| Knobs {
            coarse: match std::env::var("PROBE_COARSE").as_deref() {
                Err(_) | Ok("") | Ok("off") => Coarse::Off,
                Ok("climb") => Coarse::Climb,
                Ok("fixed") => Coarse::Fixed,
                Ok(other) => panic!("PROBE_COARSE={other}: want climb or fixed"),
            },
            bottom: match std::env::var("PROBE_BOTTOM").as_deref() {
                Err(_) | Ok("") | Ok("diagonal") => Bottom::Diagonal,
                Ok("block") => Bottom::Block,
                Ok(other) => panic!("PROBE_BOTTOM={other}: want block"),
            },
            stall1: stall("PROBE_STALL1"),
            stall2: stall("PROBE_STALL2"),
        })
    }

    /// Whether the probe asked for the coarse rung, so its build reports to stderr.
    pub(crate) fn verbose() -> bool {
        knobs().coarse != Coarse::Off
    }
}
