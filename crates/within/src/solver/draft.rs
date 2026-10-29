//! #427 + #365 type draft: the preconditioner ladder as the solver's state. Signatures only.
#![allow(dead_code, unused_variables)]

use std::time::Instant;

use once_cell::sync::OnceCell;
use schwarz_precond::draft::Par;

use super::{RhsSolution, Run};
use crate::config::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, Staleness};
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{Preconditioner, PreconditionerBuild};
use crate::{BuildError, BuildWarning, WithinError};
use schwarz_precond::SolveError;

/// The maps a solve may run on: `base`, then each rung once the map below it stalls.
/// `Off`, `Diagonal`, `Additive` and a reused map are ladders without rungs.
pub(crate) struct Ladder {
    base: PreconditionerBuild,
    rungs: Vec<Rung>,
}

/// One escalation: the stall that triggers it and the map it builds, at most once per solver.
pub(crate) struct Rung {
    stall: Staleness,
    local_solver: LocalSolverConfig,
    built: OnceCell<Result<PreconditionerBuild, BuildError>>,
}

impl Rung {
    /// The map this rung adds (`None`: no factor pair), built on the first call with all of `par`.
    fn build(
        &self,
        prepared: &PreparedDesign<'_>,
        par: Par,
    ) -> Result<Option<&Preconditioner>, BuildError> {
        todo!()
    }
}

impl Ladder {
    /// `Adaptive` gets one rung, dropped up front on a one-factor design (no factor pair).
    pub(crate) fn build(
        prepared: &PreparedDesign<'_>,
        config: PreconditionerConfig,
        par: Par,
    ) -> Result<Self, BuildError> {
        todo!()
    }

    /// A reused map is fixed: no rungs.
    pub(crate) fn reuse(preconditioner: Preconditioner) -> Self {
        Self {
            base: PreconditionerBuild {
                map: Some(preconditioner),
                warnings: Vec::new(),
            },
            rungs: Vec::new(),
        }
    }

    /// `base`, then each rung's build up to the first rung not built.
    fn builds(&self) -> impl Iterator<Item = &PreconditionerBuild> {
        std::iter::once(&self.base).chain(
            self.rungs
                .iter()
                .map_while(|r| r.built.get()?.as_ref().ok()),
        )
    }

    /// The highest built map; where the next solve starts.
    pub(crate) fn current(&self) -> Option<&Preconditioner> {
        self.builds().filter_map(|l| l.map.as_ref()).last()
    }

    /// Whether any rung has built a map.
    pub(crate) fn has_escalated(&self) -> bool {
        self.builds().skip(1).any(|l| l.map.is_some())
    }

    pub(crate) fn warnings(&self) -> impl Iterator<Item = &BuildWarning> {
        self.builds().flat_map(|l| &l.warnings)
    }

    /// Every RHS on the current map in `min(k, par)` lanes, each lane reusing one scratch
    /// buffer; per stall, build the next rung once with all of `par`, then resume the
    /// stalled RHS warm-started on it.
    pub(crate) fn solve(
        &self,
        prepared: &PreparedDesign<'_>,
        ys: &[&[f64]],
        lsmr: &LsmrOptions,
        par: Par,
    ) -> Result<(Vec<RhsSolution>, f64), WithinError> {
        // One pass over the cells: a concurrent build must not move `map` and `next` apart.
        let (mut map, mut built) = (None, 0);
        for build in self.builds() {
            (map, built) = (build.map.as_ref().or(map), built + 1);
        }
        let mut next = self.rungs[built - 1..].iter();
        let (mut done, mut build_secs) = (Vec::new(), 0.0);
        let mut pending: Vec<_> = (0..ys.len()).map(|i| (i, None)).collect();
        loop {
            let rung = next.next();
            let stall = rung.map(|r| &r.stall);
            let (finished, stalled) = run_lanes(prepared, map, stall, ys, pending, lsmr, par)?;
            done.extend(finished);
            let Some(rung) = rung.filter(|_| !stalled.is_empty()) else {
                break;
            };
            let t_build = Instant::now();
            map = rung.build(prepared, par)?.or(map);
            build_secs += t_build.elapsed().as_secs_f64();
            pending = stalled.into_iter().map(|(i, run)| (i, Some(run))).collect();
        }
        done.sort_unstable_by_key(|&(i, _)| i);
        Ok((done.into_iter().map(|(_, s)| s).collect(), build_secs))
    }
}

type Solved = (usize, RhsSolution);
type Stalled = (usize, Run);

/// `(finished, stalled)` over `(index into ys, run to resume)`; a lane gathers, runs and finishes
/// each RHS; stalled only under `stall` with iterations left; counts and seconds include the resume.
fn run_lanes(
    prepared: &PreparedDesign<'_>,
    map: Option<&Preconditioner>,
    stall: Option<&Staleness>,
    ys: &[&[f64]],
    pending: Vec<(usize, Option<Run>)>,
    lsmr: &LsmrOptions,
    par: Par,
) -> Result<(Vec<Solved>, Vec<Stalled>), SolveError> {
    todo!()
}
