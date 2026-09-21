//! The solver's preconditioner slot: a fixed map, or the diagonal→Schwarz ladder that
//! [`PreconditionerConfig::Adaptive`] builds on a stalled solve.

use std::sync::OnceLock;
use std::time::Instant;

use schwarz_precond::Staleness;

use crate::config::PreconditionerConfig;
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{build_diagonal, build_schwarz, Preconditioner, SchwarzConfig};
use crate::{BuildError, BuildWarning};

/// The solver's preconditioner: a fixed map, or an adaptive diagonal→Schwarz ladder.
pub(super) enum PrecondSlot {
    /// A single map (`None` = unpreconditioned) built at construction.
    Static(Option<Preconditioner>),
    /// Diagonal now, Schwarz built lazily on a stalled contraction.
    Adaptive(Box<AdaptivePrecond>),
}

impl PrecondSlot {
    /// Resolve a strategy: a fixed map built now, or a diagonal base with Schwarz deferred to a stall.
    pub(super) fn build(
        prepared: &PreparedDesign<'_>,
        config: PreconditionerConfig,
    ) -> Result<(Self, Vec<BuildWarning>), BuildError> {
        Ok(match config {
            PreconditionerConfig::Off => (Self::Static(None), Vec::new()),
            PreconditionerConfig::Diagonal => {
                (Self::Static(Some(build_diagonal(prepared)?)), Vec::new())
            }
            PreconditionerConfig::Additive {
                local_solver,
                reduction,
            } => {
                let schwarz = SchwarzConfig {
                    local_solver,
                    reduction,
                };
                let (map, warnings) = build_schwarz(prepared, &schwarz)?;
                (Self::Static(map), warnings)
            }
            PreconditionerConfig::Adaptive {
                local_solver,
                reduction,
                stall,
            } => {
                // The escalated build is deferred, so nothing else checks its local solver here.
                local_solver.validate()?;
                let base = build_diagonal(prepared)?;
                // One term has no factor pair to escalate to; settle now and skip the probe.
                let slot = if prepared.design.n_factors() < 2 {
                    Self::Static(Some(base))
                } else {
                    Self::Adaptive(Box::new(AdaptivePrecond {
                        base,
                        stall,
                        escalated: SchwarzConfig {
                            local_solver,
                            reduction,
                        },
                        built: OnceLock::new(),
                    }))
                };
                (slot, Vec::new())
            }
        })
    }
}

/// Diagonal-first strategy holding everything needed to build the Schwarz rung on demand.
pub(super) struct AdaptivePrecond {
    pub(super) base: Preconditioner,
    pub(super) stall: Staleness,
    /// The map built on escalation; `stall` is a solve concern it never sees.
    pub(super) escalated: SchwarzConfig,
    /// A failed build is kept too, so later solves report it instead of paying for it again.
    pub(super) built: OnceLock<Result<AdaptiveBuild, BuildError>>,
}

/// Outcome of the deferred build: the Schwarz map, or `None` when no factor-pair target exists.
pub(super) struct AdaptiveBuild {
    pub(super) schwarz: Option<Preconditioner>,
    /// Design screening followed by the deferred build, so it can stand in for `Solver::warnings`.
    pub(super) warnings: Vec<BuildWarning>,
}

impl AdaptivePrecond {
    pub(super) fn build(&self) -> Option<&AdaptiveBuild> {
        self.built.get().and_then(|b| b.as_ref().ok())
    }

    pub(super) fn schwarz(&self) -> Option<&Preconditioner> {
        self.build().and_then(|b| b.schwarz.as_ref())
    }

    /// The rung a solve should run on: the escalated Schwarz map once built, else the diagonal.
    pub(super) fn rung(&self) -> &Preconditioner {
        self.schwarz().unwrap_or(&self.base)
    }

    /// Build once and return this call's build seconds; every other caller waits for it.
    pub(super) fn escalate(
        &self,
        prepared: &PreparedDesign<'_>,
        screening: &[BuildWarning],
    ) -> Result<f64, BuildError> {
        let mut build_secs = 0.0;
        let built = self.built.get_or_init(|| {
            let t_build = Instant::now();
            let outcome = isolated(|| build_schwarz(prepared, &self.escalated))
                .and_then(|built| built)
                .map(|(schwarz, build_warnings)| {
                    let schwarz = schwarz.map(|mut p| {
                        p.gauge = self.base.gauge.clone();
                        p
                    });
                    let mut warnings = screening.to_vec();
                    warnings.extend(build_warnings);
                    AdaptiveBuild { schwarz, warnings }
                });
            build_secs = t_build.elapsed().as_secs_f64();
            outcome
        });
        built.as_ref().map(|_| build_secs).map_err(Clone::clone)
    }
}

/// Run `f` on a pool of its own, entered from a thread outside every pool, so no wait inside it can
/// steal a solve off the shared pool; a stolen solve blocking on this build is the #371 deadlock.
/// Nothing run here may wait on the shared pool.
fn isolated<R: Send>(f: impl FnOnce() -> R + Send) -> Result<R, BuildError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .build()
        .map_err(|e| BuildError::ThreadPool(e.to_string()))?;
    std::thread::scope(|s| match s.spawn(|| pool.install(f)).join() {
        Ok(r) => Ok(r),
        // Carry the build's own panic, not `Any { .. }` from formatting the payload.
        Err(payload) => std::panic::resume_unwind(payload),
    })
}

