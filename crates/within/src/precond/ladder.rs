//! The solver's preconditioner ladder: the rungs of its configuration, each built at most once,
//! the first at construction and each later one on a stall of the rung below.

use std::time::{Duration, Instant};

use once_cell::sync::OnceCell;

use crate::config::{PreconditionerConfig, Rung, Staleness};
use crate::precond::{build_map, Preconditioner};
use crate::system::Gramian;
use crate::{BuildError, BuildWarning};

/// A rung's settled build: its map, or `None` where it has nothing to act on.
type Built = Result<RungBuild, BuildError>;

struct RungBuild {
    map: Option<Preconditioner>,
    warnings: Vec<BuildWarning>,
}

/// The rungs of one configuration on one system, climbed upward only.
pub(crate) struct Ladder {
    config: PreconditionerConfig,
    /// One cell per rung; a rung the system does not offer settles without a map at construction,
    /// a design's own build failure settles too, a failed isolation pool retries.
    built: Vec<OnceCell<Built>>,
}

/// The rung a solve runs on, and the stall that hands it to the next.
#[derive(Clone, Copy)]
pub(crate) struct Current<'l> {
    top: Option<(usize, &'l Preconditioner)>,
    pub(crate) stall: Option<&'l Staleness>,
}

impl<'l> Current<'l> {
    pub(crate) fn map(&self) -> Option<&'l Preconditioner> {
        self.top.map(|(_, map)| map)
    }
}

impl Ladder {
    /// Build the first rung now; the rest wait for a stall.
    pub(crate) fn new(
        gramian: Gramian<'_>,
        config: PreconditionerConfig,
    ) -> Result<Self, BuildError> {
        // Fail before the first build, so a deferred rung never meets an invalid config.
        config.validate()?;
        let ladder = Self::unbuilt(config);
        for (at, rung) in ladder.config.rungs().enumerate() {
            let settled = if !offers(gramian, rung) {
                ladder.declined()
            } else if at == 0 {
                ladder.build(gramian, at, rung)?
            } else {
                continue;
            };
            let _ = ladder.built[at].set(Ok(settled));
        }
        Ok(ladder)
    }

    /// Resume the ladder of a map built elsewhere at the rung it was built on.
    pub(crate) fn resume(gramian: Gramian<'_>, map: Preconditioner) -> Result<Self, BuildError> {
        let n = gramian.unknowns().len();
        if map.nrows() != n || map.ncols() != n {
            return Err(BuildError::PreconditionerDimensionMismatch {
                expected: n,
                actual_rows: map.nrows(),
                actual_cols: map.ncols(),
            });
        }
        // A deserialized map skipped its config's construction-time check.
        map.config.validate()?;
        let ladder = Self::unbuilt(map.config.clone());
        let at = map.rung;
        let _ = ladder.built[at].set(Ok(RungBuild {
            map: Some(map),
            warnings: Vec::new(),
        }));
        for (above, rung) in ladder.config.rungs().enumerate().skip(at + 1) {
            if !offers(gramian, rung) {
                let _ = ladder.built[above].set(Ok(ladder.declined()));
            }
        }
        Ok(ladder)
    }

    fn unbuilt(config: PreconditionerConfig) -> Self {
        let built = config.rungs().map(|_| OnceCell::new()).collect();
        Self { config, built }
    }

    /// A rung settled without a map, carrying the warnings below it.
    fn declined(&self) -> RungBuild {
        RungBuild {
            map: None,
            warnings: self.warnings().to_vec(),
        }
    }

    fn build(&self, gramian: Gramian<'_>, at: usize, rung: Rung<'_>) -> Built {
        let started = Instant::now();
        let (map, own) = build_map(gramian, rung)?;
        // Each build carries the warnings below it, so `warnings()` only ever grows.
        let mut warnings = self.warnings().to_vec();
        warnings.extend(own);
        Ok(RungBuild {
            map: map.map(|map| Preconditioner {
                config: self.config.clone(),
                rung: at,
                map,
                build_duration: started.elapsed(),
            }),
            warnings,
        })
    }

    /// The highest rung with a map.
    fn top(&self) -> Option<(usize, &Preconditioner)> {
        self.built
            .iter()
            .enumerate()
            .rev()
            .find_map(|(at, cell)| match cell.get() {
                Some(Ok(RungBuild { map: Some(map), .. })) => Some((at, map)),
                _ => None,
            })
    }

    /// The first rung above `at` that has not settled without a map.
    fn next(&self, at: Option<usize>) -> Option<(usize, Rung<'_>)> {
        self.config
            .rungs()
            .enumerate()
            .skip(at.map_or(0, |at| at + 1))
            .find(|(at, _)| !matches!(self.built[*at].get(), Some(Ok(RungBuild { map: None, .. }))))
    }

    /// The top map, where a solve starts.
    pub(crate) fn current(&self) -> Result<Current<'_>, BuildError> {
        self.on(self.top())
    }

    /// `top`'s map, with the stall that leaves it armed only while a rung above can still be built.
    fn on<'l>(
        &'l self,
        top: Option<(usize, &'l Preconditioner)>,
    ) -> Result<Current<'l>, BuildError> {
        let at = top.map(|(at, _)| at);
        let stall = match (at, self.next(at)) {
            (Some(at), Some((next, _))) => {
                // A settled build failure is final: every later solve reports it.
                if let Some(Err(e)) = self.built[next].get() {
                    return Err(e.clone());
                }
                self.config.stall(at)
            }
            _ => None,
        };
        Ok(Current { top, stall })
    }

    /// The rung above the one `stalled` ran on, built once by whichever solve stalls first, and
    /// this call's build time. A solve climbs one rung at a time, whatever its siblings built.
    pub(crate) fn above<'l>(
        &'l self,
        gramian: Gramian<'_>,
        stalled: &Current<'l>,
    ) -> Result<(Current<'l>, Duration), BuildError> {
        let Some((next, rung)) = self.next(stalled.top.map(|(at, _)| at)) else {
            return Ok((self.on(stalled.top)?, Duration::ZERO));
        };
        let mut spent = Duration::ZERO;
        let built = self.built[next].get_or_try_init(|| {
            let started = Instant::now();
            let built = isolated(|| self.build(gramian, next, rung))?;
            spent = started.elapsed();
            Ok::<_, BuildError>(built)
        })?;
        let top = match built {
            Ok(RungBuild { map: Some(map), .. }) => Some((next, map)),
            Ok(RungBuild { map: None, .. }) => stalled.top,
            Err(e) => return Err(e.clone()),
        };
        Ok((self.on(top)?, spent))
    }

    /// The map a solve starts on: the highest rung built so far.
    pub(crate) fn preconditioner(&self) -> Option<&Preconditioner> {
        self.top().map(|(_, map)| map)
    }

    /// Whether this solver has built a rung above the one it started on.
    pub(crate) fn has_escalated(&self) -> bool {
        let start = self.built.iter().position(|cell| cell.get().is_some());
        self.top().map(|(at, _)| at) > start
    }

    /// Every settled rung's warnings, in rung order.
    pub(crate) fn warnings(&self) -> &[BuildWarning] {
        self.built
            .iter()
            .rev()
            .find_map(|cell| cell.get()?.as_ref().ok())
            .map_or(&[], |build| &build.warnings)
    }
}

/// Whether `rung` can act on this system: Schwarz needs a channel pair, hence two terms.
fn offers(gramian: Gramian<'_>, rung: Rung<'_>) -> bool {
    match rung {
        Rung::Diagonal => true,
        Rung::Schwarz(_) => gramian.layout().n_terms() >= 2,
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
