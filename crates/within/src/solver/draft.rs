//! #427 + #365 type draft: the preconditioner ladder as the solver's state. Signatures only.
#![allow(dead_code, unused_variables)]

use once_cell::sync::OnceCell;
use schwarz_precond::draft::Par;

use super::{PreparedRhs, RhsSolution};
use crate::config::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, Staleness};
use crate::domain::PreparedDesign;
use crate::operator::schwarz::{Preconditioner, PreconditionerBuild};
use crate::{BuildError, BuildWarning, WithinError};

/// The maps a solve may run on: `base`, then each rung once the previous level stalls.
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
    /// The map, built on the first call with all of `par`; the seconds this call spent building.
    fn build(
        &self,
        prepared: &PreparedDesign<'_>,
        par: Par,
    ) -> Result<(&PreconditionerBuild, f64), BuildError> {
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

    /// `base`, then each rung up to the first one not built.
    fn levels(&self) -> impl Iterator<Item = &PreconditionerBuild> {
        std::iter::once(&self.base).chain(
            self.rungs
                .iter()
                .map_while(|r| r.built.get()?.as_ref().ok()),
        )
    }

    /// The highest built map; where the next solve starts.
    pub(crate) fn current(&self) -> Option<&Preconditioner> {
        self.levels().filter_map(|l| l.map.as_ref()).last()
    }

    /// Whether any rung has built a map.
    pub(crate) fn has_escalated(&self) -> bool {
        self.levels().skip(1).any(|l| l.map.is_some())
    }

    pub(crate) fn warnings(&self) -> impl Iterator<Item = &BuildWarning> {
        self.levels().flat_map(|l| &l.warnings)
    }

    /// Every RHS on the current level in `par.split(k)` lanes, each lane reusing one scratch
    /// buffer; per stalled level, build the next rung once with all of `par`, then resume the
    /// stalled RHS warm-started on it.
    pub(crate) fn solve(
        &self,
        prepared: &PreparedDesign<'_>,
        rhs: Vec<PreparedRhs<'_>>,
        lsmr: &LsmrOptions,
        par: Par,
    ) -> Result<(Vec<RhsSolution>, f64), WithinError> {
        todo!()
    }
}
