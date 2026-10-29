//! #427 + #365 type draft: the preconditioner ladder as the solver's state. Signatures only.
#![allow(dead_code, unused_variables)]

use std::time::Instant;

use once_cell::sync::OnceCell;
use schwarz_precond::draft::Par;

use super::ladder::isolated;
use super::{RhsSolution, Run};
use crate::config::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, Staleness};
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{
    build_adaptive, build_diagonal, build_schwarz, Preconditioner, PreconditionerBuild,
};
use crate::{BuildError, BuildWarning, WithinError};
use schwarz_precond::SolveError;

/// The design and the maps a solve may run on: `base`, then each rung once the map below it
/// stalls. Only an `Adaptive` config on two or more terms has a rung.
pub(crate) struct Solver<'a> {
    prepared: PreparedDesign<'a>,
    base: PreconditionerBuild,
    rungs: Vec<Rung>,
}

/// One escalation: the stall that triggers it and the map it builds, at most once per solver.
pub(crate) struct Rung {
    stall: Staleness,
    local_solver: LocalSolverConfig,
    built: OnceCell<Result<PreconditionerBuild, BuildError>>,
}

impl<'a> Solver<'a> {
    /// `Adaptive` goes through `reuse`, the one place a rung is derived.
    pub(crate) fn build(
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
    pub(crate) fn reuse(
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
    fn builds(&self) -> impl Iterator<Item = &PreconditionerBuild> {
        std::iter::once(&self.base).chain(
            self.rungs
                .iter()
                .map_while(|r| r.built.get()?.as_ref().ok()),
        )
    }

    /// The highest built map; where the next solve starts.
    pub(crate) fn preconditioner(&self) -> Option<&Preconditioner> {
        self.builds().filter_map(|l| l.map.as_ref()).last()
    }

    /// Whether any rung has built a map.
    pub(crate) fn has_escalated(&self) -> bool {
        self.builds().skip(1).any(|l| l.map.is_some())
    }

    pub(crate) fn warnings(&self) -> Vec<BuildWarning> {
        self.builds().flat_map(|l| &l.warnings).cloned().collect()
    }

    /// Every RHS on the current map in `min(k, par)` lanes, each lane reusing one scratch
    /// buffer; per stall, build the next rung once with all of `par`, then resume the
    /// stalled RHS warm-started on it.
    pub(super) fn solve_all(
        &self,
        ys: &[&[f64]],
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
            let built = rung.built.get_or_try_init(|| {
                isolated(par, || build_schwarz(prepared, &rung.local_solver))
            })?;
            map = built.as_ref().map_err(Clone::clone)?.map.as_ref().or(map);
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
