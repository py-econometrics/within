//! The solver's preconditioner slot: a fixed map, or the diagonal→Schwarz ladder that
//! [`PreconditionerConfig::Adaptive`] builds on a stalled solve.

use std::time::Instant;

use once_cell::sync::OnceCell;

use crate::config::PreconditionerConfig;
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{
    build_adaptive, build_diagonal, build_schwarz, AdaptiveLadder, Preconditioner, SchwarzConfig,
};
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
                // Fail before the diagonal pass; `reuse` repeats this for a deserialized ladder.
                local_solver.validate()?;
                let escalated = SchwarzConfig {
                    local_solver,
                    reduction,
                };
                let base = build_adaptive(prepared, stall, escalated)?;
                (Self::reuse(prepared, base)?, Vec::new())
            }
        })
    }

    /// An unescalated ladder resumes, or settles on one term; any other map stays fixed.
    pub(super) fn reuse(
        prepared: &PreparedDesign<'_>,
        preconditioner: Preconditioner,
    ) -> Result<Self, BuildError> {
        let Some(ladder) = preconditioner.ladder() else {
            return Ok(Self::Static(Some(preconditioner)));
        };
        // A deserialized ladder skipped `Adaptive`'s build-time check, and a one-term design settles.
        ladder.escalated.local_solver.validate()?;
        Ok(if prepared.design.n_factors() < 2 {
            Self::Static(Some(preconditioner.settle()))
        } else {
            Self::Adaptive(Box::new(AdaptivePrecond {
                base: preconditioner,
                built: OnceCell::new(),
            }))
        })
    }
}

/// Diagonal-first strategy holding everything needed to build the Schwarz rung on demand.
pub(super) struct AdaptivePrecond {
    /// Applies the diagonal and carries the ladder, so handing it out keeps the strategy.
    pub(super) base: Preconditioner,
    /// A design's own build failure settles here too; a failed isolation pool retries instead.
    pub(super) built: OnceCell<Result<AdaptiveBuild, BuildError>>,
}

/// Outcome of the deferred build: the Schwarz map, or `None` when no factor-pair target exists.
pub(super) struct AdaptiveBuild {
    pub(super) schwarz: Option<Preconditioner>,
    /// Design screening followed by the deferred build, so it can stand in for `Solver::warnings`.
    pub(super) warnings: Vec<BuildWarning>,
}

impl AdaptivePrecond {
    pub(super) fn ladder(&self) -> &AdaptiveLadder {
        self.base
            .ladder()
            .expect("an adaptive slot is built only from a ladder")
    }

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
        let built = self.built.get_or_try_init(|| {
            let t_build = Instant::now();
            let outcome = isolated(|| build_schwarz(prepared, &self.ladder().escalated))?.map(
                |(schwarz, build_warnings)| {
                    let schwarz = schwarz.map(|mut p| {
                        p.gauge = self.base.gauge.clone();
                        p
                    });
                    let mut warnings = screening.to_vec();
                    warnings.extend(build_warnings);
                    AdaptiveBuild { schwarz, warnings }
                },
            );
            build_secs = t_build.elapsed().as_secs_f64();
            Ok(outcome)
        })?;
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
