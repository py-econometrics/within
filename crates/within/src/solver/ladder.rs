//! The solver's preconditioner slot: a fixed map, or the diagonal→Schwarz ladder that
//! [`PreconditionerConfig::Adaptive`] prepares speculatively and factors on a stalled solve.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use once_cell::sync::OnceCell;

use crate::build_control::{BuildContext, BuildControl, BuildFailure};
use crate::config::PreconditionerConfig;
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{
    build_adaptive, build_diagonal, build_schwarz, build_schwarz_controlled, AdaptiveLadder,
    Preconditioner, SchwarzConfig,
};
use crate::{BuildError, BuildWarning};

/// The solver's preconditioner: a fixed map, or an adaptive diagonal→Schwarz ladder.
pub(super) enum PrecondSlot {
    /// A single map (`None` = unpreconditioned) built at construction.
    Static(Option<Preconditioner>),
    /// Diagonal now, Schwarz factorization deferred to a stalled contraction.
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
                diagonal_won: AtomicBool::new(false),
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
    /// A completed diagonal batch needs no further speculative builds; later RHS still probe.
    diagonal_won: AtomicBool,
}

/// Outcome of the deferred build: the Schwarz map, or `None` when no factor-pair target exists.
pub(super) struct AdaptiveBuild {
    pub(super) schwarz: Option<Preconditioner>,
    /// Design screening followed by the deferred build, so it can stand in for `Solver::warnings`.
    pub(super) warnings: Vec<BuildWarning>,
}

/// A speculative build is published only if a solve actually needs the second rung.
pub(super) struct CandidateBuild {
    outcome: CandidateOutcome,
    pub(super) elapsed_secs: f64,
}

pub(super) enum Speculation {
    Completed(CandidateBuild),
    Cancelled { elapsed_secs: f64 },
}

impl Speculation {
    pub(super) fn elapsed_secs(&self) -> f64 {
        match self {
            Self::Completed(candidate) => candidate.elapsed_secs,
            Self::Cancelled { elapsed_secs } => *elapsed_secs,
        }
    }
}

enum CandidateOutcome {
    Ready(Result<AdaptiveBuild, BuildError>),
    /// Isolation failures are retried by synchronous escalation, never cached.
    Unavailable(BuildError),
    Panicked(Box<dyn std::any::Any + Send>),
}

impl CandidateBuild {
    pub(super) fn unavailable(error: BuildError) -> Self {
        Self {
            outcome: CandidateOutcome::Unavailable(error),
            elapsed_secs: 0.0,
        }
    }

    pub(super) fn panicked(payload: Box<dyn std::any::Any + Send>, elapsed_secs: f64) -> Self {
        Self {
            outcome: CandidateOutcome::Panicked(payload),
            elapsed_secs,
        }
    }
}

impl AdaptivePrecond {
    pub(super) fn should_speculate(&self) -> bool {
        !self.diagonal_won.load(Ordering::Acquire)
    }

    pub(super) fn keep_diagonal(&self) {
        self.diagonal_won.store(true, Ordering::Release);
    }

    /// Prepare against the existing design inside the pool entered by `spawn_isolated`.
    pub(super) fn speculate(
        &self,
        prepared: &PreparedDesign<'_>,
        screening: &[BuildWarning],
        control: &BuildControl,
    ) -> Speculation {
        let started = Instant::now();
        let outcome = build_schwarz_controlled(
            prepared,
            &self.ladder().escalated,
            BuildContext(Some(control)),
        );
        let outcome = match outcome {
            Ok((schwarz, warnings)) => {
                CandidateOutcome::Ready(Ok(self.assembled(schwarz, warnings, screening)))
            }
            Err(BuildFailure::Failed(error)) => CandidateOutcome::Ready(Err(error)),
            Err(BuildFailure::Cancelled) => {
                return Speculation::Cancelled {
                    elapsed_secs: started.elapsed().as_secs_f64(),
                };
            }
        };
        Speculation::Completed(CandidateBuild {
            outcome,
            elapsed_secs: started.elapsed().as_secs_f64(),
        })
    }

    /// The first concurrent caller to publish wins; build errors remain cached as before.
    pub(super) fn publish(
        &self,
        candidate: CandidateBuild,
        prepared: &PreparedDesign<'_>,
        screening: &[BuildWarning],
    ) -> Result<f64, BuildError> {
        let outcome = match candidate.outcome {
            CandidateOutcome::Ready(outcome) => outcome,
            CandidateOutcome::Unavailable(_error) => return self.escalate(prepared, screening),
            CandidateOutcome::Panicked(payload) => std::panic::resume_unwind(payload),
        };
        self.built
            .get_or_init(|| outcome)
            .as_ref()
            .map(|_| 0.0)
            .map_err(Clone::clone)
    }

    fn assembled(
        &self,
        schwarz: Option<Preconditioner>,
        build_warnings: Vec<BuildWarning>,
        screening: &[BuildWarning],
    ) -> AdaptiveBuild {
        let schwarz = schwarz.map(|mut p| {
            p.gauge = self.base.gauge.clone();
            p
        });
        let mut warnings = screening.to_vec();
        warnings.extend(build_warnings);
        AdaptiveBuild { schwarz, warnings }
    }

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
            let outcome = isolated(|| build_schwarz(prepared, &self.ladder().escalated))?
                .map(|(schwarz, warnings)| self.assembled(schwarz, warnings, screening));
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
    let threads = rayon::current_num_threads();
    std::thread::scope(|s| {
        let bridge = spawn_isolated(s, threads, f)?;
        match bridge.join() {
            Ok(r) => r,
            // Carry the build's own panic, not `Any { .. }` from formatting the payload.
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

/// Enter an isolated pool only from this bridge thread, outside every Rayon pool.
/// The closure must never wait for work on the caller's pool (#371).
pub(super) fn spawn_isolated<'scope, 'env, R: Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    threads: usize,
    f: impl FnOnce() -> R + Send + 'scope,
) -> Result<std::thread::ScopedJoinHandle<'scope, Result<R, BuildError>>, BuildError> {
    std::thread::Builder::new()
        .name("within-schwarz".into())
        .spawn_scoped(scope, move || on_pool(threads, f))
        .map_err(|e| BuildError::ThreadPool(e.to_string()))
}

fn on_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> Result<R, BuildError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| BuildError::ThreadPool(e.to_string()))?;
    Ok(pool.install(f))
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
