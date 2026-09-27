//! The solver's preconditioner slot: a fixed map, or the diagonal→Schwarz ladder that
//! [`PreconditionerConfig::Adaptive`] builds on a stalled solve.

use std::time::Instant;

use once_cell::sync::OnceCell;

use crate::config::{PreconditionerConfig, Staleness};
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{
    build_adaptive, build_diagonal, build_schwarz, Preconditioner, SchwarzConfig,
};
use crate::{BuildError, BuildWarning};

/// The solver's preconditioner: a fixed map, or an adaptive diagonal→Schwarz ladder.
pub(super) enum PrecondSlot {
    /// A single map (`None` = unpreconditioned) built at construction.
    Static {
        map: Option<Preconditioner>,
        /// Design screening followed by the map's build.
        warnings: Vec<BuildWarning>,
    },
    /// Diagonal now, Schwarz built lazily on a stalled contraction.
    Adaptive(Box<AdaptivePrecond>),
}

/// The map a solve starts on: a settled one, or the ladder's diagonal probing for a stall.
pub(super) enum Active<'s> {
    Fixed(Option<&'s Preconditioner>),
    Probe(&'s AdaptivePrecond),
}

impl PrecondSlot {
    /// Resolve a strategy: a fixed map built now, or a diagonal base with Schwarz deferred to a stall.
    pub(super) fn build(
        prepared: &PreparedDesign<'_>,
        config: PreconditionerConfig,
        mut warnings: Vec<BuildWarning>,
    ) -> Result<Self, BuildError> {
        let map = match config {
            PreconditionerConfig::Off => None,
            PreconditionerConfig::Diagonal => Some(build_diagonal(prepared)?),
            PreconditionerConfig::Additive {
                local_solver,
                reduction,
            } => {
                let schwarz = SchwarzConfig {
                    local_solver,
                    reduction,
                };
                let (map, build_warnings) = build_schwarz(prepared, &schwarz)?;
                warnings.extend(build_warnings);
                map
            }
            PreconditionerConfig::Adaptive {
                local_solver,
                reduction,
                stall,
            } => {
                // Fail before the diagonal pass; `reuse` repeats this for a deserialized ladder.
                local_solver.validate()?;
                let escalated = SchwarzConfig {
                    local_solver,
                    reduction,
                };
                let base = build_adaptive(prepared, stall, escalated)?;
                return Self::reuse(prepared, base, warnings);
            }
        };
        Ok(Self::Static { map, warnings })
    }

    /// An unescalated ladder resumes, or settles on one term; any other map stays fixed.
    pub(super) fn reuse(
        prepared: &PreparedDesign<'_>,
        preconditioner: Preconditioner,
        warnings: Vec<BuildWarning>,
    ) -> Result<Self, BuildError> {
        if let Some(ladder) = preconditioner.ladder() {
            // A deserialized ladder skipped `Adaptive`'s build-time check, and a one-term design settles.
            ladder.escalated.local_solver.validate()?;
            if prepared.design.n_factors() >= 2 {
                let (stall, escalated) = (ladder.stall, ladder.escalated.clone());
                return Ok(Self::Adaptive(Box::new(AdaptivePrecond {
                    base: preconditioner,
                    stall,
                    escalated,
                    screening: warnings,
                    built: OnceCell::new(),
                })));
            }
        }
        let map = Some(preconditioner.settle());
        Ok(Self::Static { map, warnings })
    }

    /// What a solve starts on; a settled build error is final.
    pub(super) fn active(&self) -> Result<Active<'_>, &BuildError> {
        Ok(match self {
            Self::Static { map, .. } => Active::Fixed(map.as_ref()),
            Self::Adaptive(a) => match a.built.get() {
                None => Active::Probe(a),
                Some(settled) => Active::Fixed(Some(a.rung(settled)?)),
            },
        })
    }

    /// The persistable map: under Adaptive, the Schwarz map once built, else the ladder itself.
    pub(super) fn preconditioner(&self) -> Option<&Preconditioner> {
        match self {
            Self::Static { map, .. } => map.as_ref(),
            Self::Adaptive(a) => Some(match a.built.get() {
                Some(Settled::Escalated { rung, .. }) => rung,
                None | Some(Settled::NoTarget | Settled::Failed(_)) => &a.base,
            }),
        }
    }

    pub(super) fn warnings(&self) -> &[BuildWarning] {
        match self {
            Self::Static { warnings, .. } => warnings,
            Self::Adaptive(a) => match a.built.get() {
                Some(Settled::Escalated { warnings, .. }) => warnings,
                None | Some(Settled::NoTarget | Settled::Failed(_)) => &a.screening,
            },
        }
    }

    pub(super) fn has_escalated(&self) -> bool {
        matches!(self, Self::Adaptive(a) if matches!(a.built.get(), Some(Settled::Escalated { .. })))
    }
}

/// Diagonal-first strategy; its private fields keep `reuse` the only constructor.
pub(super) struct AdaptivePrecond {
    /// Applies the diagonal and carries the ladder, so handing it out keeps the strategy.
    pub(super) base: Preconditioner,
    pub(super) stall: Staleness,
    escalated: SchwarzConfig,
    screening: Vec<BuildWarning>,
    /// Empty until a build settles; a failed isolation pool leaves it empty, so the next stall retries.
    built: OnceCell<Settled>,
}

/// Outcome of the deferred build; each one is final for the solver.
enum Settled {
    /// The Schwarz rung, with the screening followed by its build warnings.
    Escalated {
        rung: Preconditioner,
        warnings: Vec<BuildWarning>,
    },
    /// No factor-pair target exists, so the diagonal stays.
    NoTarget,
    Failed(BuildError),
}

impl AdaptivePrecond {
    /// Build once and return the rung stalled solves resume on, with this call's build seconds.
    pub(super) fn escalate(
        &self,
        prepared: &PreparedDesign<'_>,
    ) -> Result<(&Preconditioner, f64), BuildError> {
        let mut build_secs = 0.0;
        let settled = self.built.get_or_try_init(|| {
            let t_build = Instant::now();
            let settled = isolated(|| match build_schwarz(prepared, &self.escalated) {
                Ok((Some(mut schwarz), build_warnings)) => Settled::Escalated {
                    rung: {
                        schwarz.gauge = self.base.gauge.clone();
                        schwarz
                    },
                    warnings: [self.screening.as_slice(), &build_warnings].concat(),
                },
                Ok((None, build_warnings)) => {
                    debug_assert!(build_warnings.is_empty(), "a warning names a subdomain");
                    Settled::NoTarget
                }
                Err(e) => Settled::Failed(e),
            })?;
            build_secs = t_build.elapsed().as_secs_f64();
            Ok::<_, BuildError>(settled)
        })?;
        let rung = self.rung(settled).map_err(Clone::clone)?;
        Ok((rung, build_secs))
    }

    /// The map a settled ladder runs on: the Schwarz rung, or the diagonal when none exists.
    fn rung<'s>(&'s self, settled: &'s Settled) -> Result<&'s Preconditioner, &'s BuildError> {
        match settled {
            Settled::Escalated { rung, .. } => Ok(rung),
            Settled::NoTarget => Ok(&self.base),
            Settled::Failed(e) => Err(e),
        }
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
    std::thread::scope(|s| {
        let bridge = std::thread::Builder::new()
            .spawn_scoped(s, || pool.install(f))
            .map_err(|e| BuildError::ThreadPool(e.to_string()))?;
        match bridge.join() {
            Ok(r) => Ok(r),
            // Carry the build's own panic, not `Any { .. }` from formatting the payload.
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use super::isolated;

    thread_local! {
        static ON_CALLER_POOL: Cell<bool> = const { Cell::new(false) };
    }

    /// A build handed to [`isolated`] runs off the caller's pool, so it can steal none of its jobs.
    #[test]
    fn isolated_leaves_the_callers_pool() {
        let caller = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .start_handler(|_| ON_CALLER_POOL.with(|f| f.set(true)))
            .build()
            .expect("caller pool");
        assert!(
            caller.install(|| ON_CALLER_POOL.with(Cell::get)),
            "the marker never reached the caller pool's workers"
        );
        assert!(
            !caller
                .install(|| isolated(|| ON_CALLER_POOL.with(Cell::get)))
                .expect("isolated build"),
            "the build ran on the caller's pool"
        );
    }

    /// A worker waiting on the build must not steal: running a queued solve above the build frame
    /// is the #371 deadlock, so entering the build pool from the worker itself is not enough.
    #[test]
    fn a_worker_waiting_on_the_build_runs_nothing_else() {
        let caller = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("caller pool");
        let ran = Arc::new(AtomicBool::new(false));
        let queued = Arc::clone(&ran);
        caller.install(move || {
            rayon::spawn(move || queued.store(true, Ordering::SeqCst));
            isolated(|| std::thread::sleep(Duration::from_millis(50))).expect("isolated build");
            assert!(
                !ran.load(Ordering::SeqCst),
                "the waiting worker stole a queued job while the build ran"
            );
        });
    }
}
