//! The solver's preconditioner slot: a [`PreconditionerConfig`] and the map of each rung built
//! so far.

use std::sync::Arc;
use std::time::Instant;

use once_cell::sync::OnceCell;
use schwarz_precond::Staleness;

use crate::config::{Handoff, PreconditionerConfig, RungSettings};
use crate::operator::schwarz::{Preconditioner, RungBuild, RungMap};
use crate::{BuildError, BuildWarning};

use super::system::{offer, Build, LeastSquaresSystem};

/// One rung's build: the map and its warnings, or the error it keeps.
type Built = Result<RungBuild<Preconditioner>, BuildError>;

/// The solver's config; a solve runs on its highest built rung and builds the next on a stall.
pub(super) struct PrecondSlot {
    config: PreconditionerConfig,
    /// One per rung of `config`, in its order. A design's own build failure settles here too;
    /// a failed isolation pool retries instead.
    pub(super) built: Vec<OnceCell<Built>>,
}

/// The map a solve runs on now, where it stands among the rungs, and the stall that hands it
/// to a rung still to be built.
pub(super) struct Running<'s> {
    at: Option<usize>,
    pub(super) map: Option<&'s Preconditioner>,
    pub(super) stall: Option<&'s Staleness>,
}

impl PrecondSlot {
    /// `config` with rungs built on `system` until a solve runs on one.
    pub(super) fn build(
        system: &impl LeastSquaresSystem,
        config: PreconditionerConfig,
    ) -> Result<Self, BuildError> {
        config.validate()?;
        let slot = Self::unbuilt(config);
        slot.settle_with(system, |build| Ok(slot.run(system, build)))?;
        Ok(slot)
    }

    /// The config `preconditioner` is a rung of, resumed from that rung, if its map acts on
    /// `system`'s coefficients.
    pub(super) fn reuse(
        system: &impl LeastSquaresSystem,
        preconditioner: Preconditioner,
    ) -> Result<Self, BuildError> {
        if let Some(mismatch) = system.layout().mismatch(&preconditioner.layout) {
            return Err(BuildError::PreconditionerMismatch(mismatch));
        }
        // A deserialized config skipped `build`'s checks.
        preconditioner.config.validate()?;
        let at = preconditioner
            .config
            .position(preconditioner.rung())
            .expect("a handed-out map is a rung of its config");
        let slot = Self::unbuilt(preconditioner.config.clone());
        slot.built[at].get_or_init(|| {
            Ok(RungBuild {
                map: Some(preconditioner),
                warnings: Vec::new(),
            })
        });
        slot.settle_with(system, |build| Ok(slot.run(system, build)))?;
        Ok(slot)
    }

    fn unbuilt(config: PreconditionerConfig) -> Self {
        let built = config.rungs().map(|_| OnceCell::new()).collect();
        Self { config, built }
    }

    fn run(
        &self,
        system: &impl LeastSquaresSystem,
        build: &Build<'_>,
    ) -> Result<RungBuild<Preconditioner>, BuildError> {
        let started = Instant::now();
        let build = build()?;
        Ok(RungBuild {
            map: build.map.map(|map| Preconditioner {
                map,
                config: self.config.clone(),
                layout: Arc::new(system.layout()),
                build_duration: started.elapsed(),
            }),
            warnings: build.warnings,
        })
    }

    /// The highest rung with a map: its position, settings and map.
    fn top(&self) -> Option<(usize, RungSettings<'_>, &Preconditioner)> {
        self.config
            .rungs()
            .zip(&self.built)
            .enumerate()
            .filter_map(|(at, (rung, cell))| match cell.get() {
                Some(Ok(RungBuild { map: Some(map), .. })) => Some((at, rung, map)),
                _ => None,
            })
            .last()
    }

    /// The map of the rung under `at`, once built.
    fn below(&self, at: usize) -> Option<&RungMap> {
        match self.built[at.checked_sub(1)?].get()? {
            Ok(RungBuild { map: Some(p), .. }) => Some(&p.map),
            _ => None,
        }
    }

    /// The first rung above `floor` that `system` offers and that was not built without a map.
    fn next<'s, S: LeastSquaresSystem>(
        &'s self,
        system: &'s S,
        floor: Option<usize>,
    ) -> Option<(usize, RungSettings<'s>, Build<'s>)> {
        self.config
            .rungs()
            .enumerate()
            .skip(floor.map_or(0, |at| at + 1))
            .filter(|&(at, _)| {
                !matches!(self.built[at].get(), Some(Ok(RungBuild { map: None, .. })))
            })
            .find_map(|(at, rung)| Some((at, rung, offer(system, rung, self.below(at))?)))
    }

    /// Build the rungs a solve must not start below; this call's build seconds. Every other
    /// caller waits for the same builds.
    pub(super) fn settle(&self, system: &impl LeastSquaresSystem) -> Result<f64, BuildError> {
        self.settle_with(system, |build| isolated(|| self.run(system, build)))
    }

    fn settle_with(
        &self,
        system: &impl LeastSquaresSystem,
        run: impl Fn(&Build<'_>) -> Result<Built, BuildError>,
    ) -> Result<f64, BuildError> {
        match self.top() {
            Some((_, rung, _)) if runs_on(&rung) => Ok(0.0),
            top => self.climb(system, top.map(|(at, ..)| at), run),
        }
    }

    /// The map a solve runs on now, once settled; a kept build error of the rung above it is
    /// final.
    pub(super) fn running(
        &self,
        system: &impl LeastSquaresSystem,
    ) -> Result<Running<'_>, BuildError> {
        let top = self.top();
        let at = top.map(|(at, ..)| at);
        let stall = match self.next(system, at) {
            Some((next, ..)) => {
                if let Some(Err(e)) = self.built[next].get() {
                    return Err(e.clone());
                }
                top.and_then(|(_, rung, _)| match rung.handoff()? {
                    Handoff::OnStall(stall) => Some(stall),
                    Handoff::AtBuild => None,
                })
            }
            None => None,
        };
        Ok(Running {
            at,
            map: top.map(|(.., map)| map),
            stall,
        })
    }

    /// Build rungs above the one `stalled` ran on until a solve runs on one; this call's build
    /// seconds. Every other caller waits for the same builds.
    pub(super) fn escalate(
        &self,
        system: &impl LeastSquaresSystem,
        stalled: &Running<'_>,
    ) -> Result<f64, BuildError> {
        self.climb(system, stalled.at, |build| {
            isolated(|| self.run(system, build))
        })
    }

    /// Build rungs above `floor`, each through `run`, until a solve runs on one; a kept build
    /// error is final; an error `run` returns without a build is retried by the next caller.
    fn climb(
        &self,
        system: &impl LeastSquaresSystem,
        mut floor: Option<usize>,
        run: impl Fn(&Build<'_>) -> Result<Built, BuildError>,
    ) -> Result<f64, BuildError> {
        let mut build_secs = 0.0;
        while let Some((at, rung, build)) = self.next(system, floor) {
            let built = self.built[at].get_or_try_init(|| {
                let started = Instant::now();
                let outcome = run(&build)?;
                build_secs += started.elapsed().as_secs_f64();
                Ok::<_, BuildError>(outcome)
            })?;
            let built = built.as_ref().map_err(Clone::clone)?;
            if built.map.is_some() && runs_on(&rung) {
                break;
            }
            floor = Some(at);
        }
        Ok(build_secs)
    }

    pub(super) fn preconditioner(&self) -> Option<&Preconditioner> {
        self.top().map(|(.., map)| map)
    }

    /// Every built rung's non-fatal events, in rung order.
    pub(super) fn warnings(&self) -> Vec<BuildWarning> {
        self.built
            .iter()
            .filter_map(|cell| cell.get()?.as_ref().ok())
            .flat_map(|built| built.warnings.iter().cloned())
            .collect()
    }
}

/// A solve runs on `rung` once it has a map: it does not hand off at build.
fn runs_on(rung: &RungSettings<'_>) -> bool {
    !matches!(rung.handoff(), Some(Handoff::AtBuild))
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
