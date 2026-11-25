//! The solver's preconditioner ladder: the rungs of its configuration, each built at most once,
//! the first at construction and each later one on a stall of the rung below.

use std::time::{Duration, Instant};

use once_cell::sync::OnceCell;
use schwarz_precond::{
    mlsmr, EscalationPolicy, IdentityOp, LsmrResult, MlsmrOptions, Operator, SolveError,
};

use crate::config::{LsmrOptions, PreconditionerConfig, Rung, Staleness};
use crate::precond::Preconditioner;
use crate::system::{Gramian, SystemMatrix};
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
    stall: Option<&'l Staleness>,
}

impl Current<'_> {
    /// LSMR on `a` from `start` within what `spent` leaves of `lsmr.maxiter`, on this rung's map
    /// (`I` without one), handing off on its stall.
    pub(crate) fn run(
        &self,
        a: &SystemMatrix<'_>,
        b: &[f64],
        lsmr: &LsmrOptions,
        spent: usize,
        start: Option<&[f64]>,
    ) -> Result<LsmrResult, SolveError> {
        let identity = IdentityOp { n: a.ncols() };
        let m: &dyn Operator = match self.top {
            Some((_, map)) => map,
            None => &identity,
        };
        let options = MlsmrOptions {
            escalation: self.stall.map(|s| s as &dyn EscalationPolicy),
            local_size: lsmr.local_size,
        };
        mlsmr(a, b, m, lsmr.tol, lsmr.maxiter - spent, start, options)
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
        if let Some((rung, _)) = ladder
            .config
            .rungs()
            .next()
            .filter(|(r, _)| r.offers(gramian))
        {
            let first = ladder.build(gramian, 0, rung)?;
            let _ = ladder.built[0].set(Ok(first));
        }
        ladder.decline_unoffered(gramian);
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
        ladder.decline_unoffered(gramian);
        Ok(ladder)
    }

    fn unbuilt(config: PreconditionerConfig) -> Self {
        let built = config.rungs().map(|_| OnceCell::new()).collect();
        Self { config, built }
    }

    /// Settle each rung the system does not offer, without a map; run after the base rung, so each
    /// carries the warnings below it.
    fn decline_unoffered(&self, gramian: Gramian<'_>) {
        for (cell, (rung, _)) in self.built.iter().zip(self.config.rungs()) {
            if !rung.offers(gramian) {
                let _ = cell.set(Ok(RungBuild {
                    map: None,
                    warnings: self.warnings().to_vec(),
                }));
            }
        }
    }

    fn build(&self, gramian: Gramian<'_>, at: usize, rung: Rung<'_>) -> Built {
        let started = Instant::now();
        let (map, own) = rung.build(gramian)?;
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
            .map(|(rung, _)| rung)
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
                self.config.rungs().nth(at).and_then(|(_, stall)| stall)
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
            // A worker stealing a solve that waits on this build is the #371 deadlock.
            let built = inflight::parallel::isolated(|| self.build(gramian, next, rung))
                .map_err(|e| BuildError::ThreadPool(e.to_string()))?;
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
