//! The maps a solve runs on, each `Adaptive` rung built on a stall, and the lanes over the RHS.

use std::time::Instant;

use once_cell::sync::OnceCell;
use schwarz_precond::par::join;
use schwarz_precond::Par;
use schwarz_precond::{
    mlsmr, IdentityPrecond, InitialGuess, LsmrOptions, LsmrStopReason, MemBuffer, StackReq,
    Staleness,
};

use super::{PreparedRhs, RhsSolution, Run, Solver};
use crate::config::{LocalSolverConfig, PreconditionerConfig};
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{
    build_adaptive, build_diagonal, build_schwarz, Preconditioner, PreconditionerBuild,
};
use crate::{BuildError, SolveError, WithinError};

/// One escalation: the stall that triggers it and the map it builds, at most once per solver.
pub(super) struct Rung {
    stall: Staleness,
    local_solver: LocalSolverConfig,
    /// A design's own build failure settles here; a failed isolation pool retries instead.
    pub(super) built: OnceCell<Result<PreconditionerBuild, BuildError>>,
}

impl<'a> Solver<'a> {
    /// `Adaptive` goes through `reuse`, the one place a rung is derived.
    pub(super) fn build(
        prepared: PreparedDesign<'a>,
        config: PreconditionerConfig,
    ) -> Result<Self, BuildError> {
        let base = match config {
            PreconditionerConfig::Off => PreconditionerBuild {
                map: None,
                warnings: Vec::new(),
            },
            PreconditionerConfig::Diagonal => PreconditionerBuild {
                map: Some(build_diagonal(&prepared)?),
                warnings: Vec::new(),
            },
            PreconditionerConfig::Additive { local_solver } => {
                build_schwarz(&prepared, &local_solver)?
            }
            PreconditionerConfig::Adaptive {
                local_solver,
                stall,
            } => {
                // Fail before the diagonal pass; `reuse` repeats this for a deserialized ladder.
                local_solver.validate()?;
                let map = build_adaptive(&prepared, stall, local_solver)?;
                return Self::reuse(prepared, map);
            }
        };
        Ok(Self {
            prepared,
            base,
            rungs: Vec::new(),
        })
    }

    /// The map as base plus its `Adaptive` rung; on one term it settles to its diagonal, no rung.
    pub(super) fn reuse(
        prepared: PreparedDesign<'a>,
        preconditioner: Preconditioner,
    ) -> Result<Self, BuildError> {
        let n_dofs = prepared.design.n_dofs;
        let (actual_rows, actual_cols) = (preconditioner.nrows(), preconditioner.ncols());
        if actual_rows != n_dofs || actual_cols != n_dofs {
            return Err(BuildError::PreconditionerDimensionMismatch {
                expected: n_dofs,
                actual_rows,
                actual_cols,
            });
        }
        let rung = match preconditioner.config() {
            PreconditionerConfig::Adaptive {
                local_solver,
                stall,
            } => {
                // A deserialized ladder skipped `Adaptive`'s build-time check.
                local_solver.validate()?;
                (prepared.design.n_factors() >= 2).then(|| Rung {
                    stall,
                    local_solver,
                    built: OnceCell::new(),
                })
            }
            _ => None,
        };
        let map = match rung {
            Some(_) => preconditioner,
            None => preconditioner.settle(),
        };
        Ok(Self {
            prepared,
            base: PreconditionerBuild {
                map: Some(map),
                warnings: Vec::new(),
            },
            rungs: rung.into_iter().collect(),
        })
    }

    /// `base`, then each rung's build up to the first rung not built.
    pub(super) fn builds(&self) -> impl Iterator<Item = &PreconditionerBuild> {
        std::iter::once(&self.base).chain(
            self.rungs
                .iter()
                .map_while(|r| r.built.get()?.as_ref().ok()),
        )
    }

    /// Every RHS in lanes; a stall builds the next rung once under all of `par`, then resumes.
    pub(super) fn solve_all<'s>(
        &'s self,
        ys: &'s [&'s [f64]],
        lsmr: &LsmrOptions,
        par: Par,
    ) -> Result<(Vec<RhsSolution>, f64), WithinError> {
        let prepared = &self.prepared;
        // One pass over the cells: a concurrent build must not move `map` and `next` apart.
        let (mut map, mut built) = (None, 0);
        for build in self.builds() {
            (map, built) = (build.map.as_ref().or(map), built + 1);
        }
        let mut next = self.rungs[built - 1..].iter();
        // A settled rung never re-probes: its kept build error is final.
        if let Some(Some(Err(e))) = self.rungs.get(built - 1).map(|r| r.built.get()) {
            return Err(e.clone().into());
        }
        let (mut done, mut build_secs) = (Vec::with_capacity(ys.len()), 0.0);
        let mut pending: Vec<_> = (0..ys.len()).map(|i| (i, None)).collect();
        loop {
            let rung = next.next();
            let stall = rung.map(|r| r.stall);
            let lanes = Lanes {
                solver: self,
                map,
                stall,
                ys,
                lsmr,
            };
            let (finished, stalled) = lanes.run(pending, par)?;
            done.extend(finished);
            let Some(rung) = rung.filter(|_| !stalled.is_empty()) else {
                break;
            };
            let t_build = Instant::now();
            let built = rung.built.get_or_try_init(|| {
                isolated(par, || build_schwarz(prepared, &rung.local_solver))
            })?;
            map = built.as_ref().map_err(Clone::clone)?.map.as_ref().or(map);
            build_secs += t_build.elapsed().as_secs_f64();
            pending = stalled
                .into_iter()
                .map(|(i, rhs, run)| (i, Some((rhs, run))))
                .collect();
        }
        done.sort_unstable_by_key(|&(i, _)| i);
        Ok((done.into_iter().map(|(_, s)| s).collect(), build_secs))
    }
}

type Solved = (usize, RhsSolution);
type Stalled<'s> = (usize, PreparedRhs<'s>, Run);
type Pending<'s> = (usize, Option<(PreparedRhs<'s>, Run)>);

/// One pass over pending RHS on one map, under `stall` when a rung waits above it.
struct Lanes<'s, 'a> {
    solver: &'s Solver<'a>,
    map: Option<&'s Preconditioner>,
    stall: Option<Staleness>,
    ys: &'s [&'s [f64]],
    lsmr: &'s LsmrOptions,
}

impl<'s> Lanes<'s, '_> {
    /// Halves `pending` and `par` together; each lane runs its RHS in order on one scratch.
    fn run(
        &self,
        mut pending: Vec<Pending<'s>>,
        par: Par,
    ) -> Result<(Vec<Solved>, Vec<Stalled<'s>>), SolveError> {
        if pending.len() <= 1 || par.degree() <= 1 {
            return self.lane(pending, par);
        }
        let right = pending.split_off(pending.len().div_ceil(2));
        let (l, r) = join(par, |p| self.run(pending, p), |p| self.run(right, p));
        let ((mut done, mut stalled), (more_done, more_stalled)) = (l?, r?);
        done.extend(more_done);
        stalled.extend(more_stalled);
        Ok((done, stalled))
    }

    fn lane(
        &self,
        pending: Vec<Pending<'s>>,
        par: Par,
    ) -> Result<(Vec<Solved>, Vec<Stalled<'s>>), SolveError> {
        let mut scratch = MemBuffer::new(StackReq::EMPTY);
        let (mut done, mut stalled) = (Vec::new(), Vec::new());
        for (i, prior) in pending {
            let (rhs, prior) = match prior {
                Some((rhs, run)) => (rhs, Some(run)),
                None => (self.solver.prepare(self.ys[i])?, None),
            };
            let run = self.resume(&rhs, prior, par, &mut scratch)?;
            // A stall on the last permitted iteration leaves the rung nothing to spend.
            if run.info.stop_reason == LsmrStopReason::Stalled
                && run.info.iterations < self.lsmr.maxiter
            {
                stalled.push((i, rhs, run));
            } else {
                done.push((i, self.solver.finish(rhs, run, par)));
            }
        }
        Ok((done, stalled))
    }

    /// One run on `rhs`, warm-started from `prior`, which spent part of the iteration budget.
    fn resume(
        &self,
        rhs: &PreparedRhs<'_>,
        prior: Option<Run>,
        par: Par,
        scratch: &mut MemBuffer,
    ) -> Result<Run, SolveError> {
        let prepared = &self.solver.prepared;
        let (mut x, start, spent, spent_secs) = match prior {
            Some(run) => (
                run.x,
                InitialGuess::Given,
                run.info.iterations,
                run.solve_secs,
            ),
            None => (
                vec![0.0; prepared.design.n_dofs],
                InitialGuess::Zero,
                0,
                0.0,
            ),
        };
        let options = LsmrOptions {
            maxiter: self.lsmr.maxiter - spent,
            ..*self.lsmr
        };
        let b = rhs.b();
        let t_start = Instant::now();
        let (xs, stall) = (&mut x[..], self.stall);
        let mut info = match self.map {
            Some(m) => mlsmr(prepared, b, m, xs, start, &options, stall, par, scratch)?,
            None => {
                let identity = IdentityPrecond {
                    dim: prepared.design.n_dofs,
                };
                mlsmr(
                    prepared, b, &identity, xs, start, &options, stall, par, scratch,
                )?
            }
        };
        info.iterations += spent;
        Ok(Run {
            x,
            info,
            solve_secs: t_start.elapsed().as_secs_f64() + spent_secs,
        })
    }
}

/// Run `f` on a pool of its own, entered from a thread outside every pool, so no wait inside it can
/// steal a solve off the shared pool; a stolen solve blocking on this build is the #371 deadlock.
/// Nothing run here may wait on the shared pool.
pub(super) fn isolated<R: Send>(par: Par, f: impl FnOnce() -> R + Send) -> Result<R, BuildError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(par.degree())
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

    use super::{isolated, Par};

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
                .install(|| isolated(Par::rayon(0), || ON_CALLER_POOL.with(Cell::get)))
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
            isolated(Par::rayon(0), || {
                std::thread::sleep(Duration::from_millis(50))
            })
            .expect("isolated build");
            assert!(
                !ran.load(Ordering::SeqCst),
                "the waiting worker stole a queued job while the build ran"
            );
        });
    }
}
