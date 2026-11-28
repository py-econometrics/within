//! `M ≈ S⁻¹`: the diagonal and Schwarz preconditioners, read from `diag(S)` and `G`'s pair
//! factors, and the opaque public [`Preconditioner`] handle.

use schwarz_precond::{MemStack, Operator, StackReq};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use crate::config::{PreconditionerConfig, Rung};
use crate::system::Gramian;
use crate::{BuildError, BuildWarning};
use schwarz::FeSchwarz;

mod ladder;
mod schwarz;
mod two_level;

pub(crate) use ladder::{Current, Ladder};
pub(crate) use two_level::Deflated;

/// Jacobi preconditioner on `diag(S)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiagonalPreconditioner {
    inv_diag: Arc<[f64]>,
}

impl Operator for DiagonalPreconditioner {
    fn nrows(&self) -> usize {
        self.inv_diag.len()
    }

    fn ncols(&self) -> usize {
        self.inv_diag.len()
    }

    fn apply_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        let n = self.inv_diag.len();
        if x.len() != n || y.len() != n {
            return Err(schwarz_precond::SolveError::InvalidInput {
                context: "DiagonalPreconditioner::apply",
                message: format!(
                    "x.len()={}, y.len()={}, expected n_dofs={}",
                    x.len(),
                    y.len(),
                    n
                ),
            });
        }
        for ((yi, &xi), &di) in y.iter_mut().zip(x.iter()).zip(self.inv_diag.iter()) {
            *yi = di * xi;
        }
        Ok(())
    }

    /// A diagonal operator is symmetric (`M^T = M`), so the adjoint is `apply`.
    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        self.apply_with(x, y, stack)
    }
}

/// A built map `M⁻¹` and the configuration whose ladder it is a rung of; cloning is O(1).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "PreconditionerWire")]
pub struct Preconditioner {
    config: PreconditionerConfig,
    /// Its position among `config`'s rungs.
    rung: usize,
    map: Map,
    build_duration: Duration,
}

/// The map a rung builds.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum Map {
    // Append only: postcard encodes by declaration order and the fixtures depend on it.
    Diagonal(DiagonalPreconditioner),
    Schwarz(FeSchwarz),
    /// PROTOTYPE(#607): Schwarz with the rigid coarse correction; never persisted.
    TwoLevel(two_level::TwoLevelMap),
}

impl Rung<'_> {
    /// Whether this rung can act on the system: Schwarz needs a pair block.
    fn offers(&self, gramian: Gramian<'_>) -> bool {
        match self {
            Self::Diagonal => true,
            Self::Schwarz(_) | Self::Coarse(_) => gramian.has_pair_blocks(),
        }
    }

    fn builds(&self, map: &Map) -> bool {
        matches!(
            (self, map),
            (Self::Diagonal, Map::Diagonal(_))
                | (Self::Schwarz(_), Map::Schwarz(_))
                | (Self::Coarse(_), Map::TwoLevel(_) | Map::Schwarz(_))
        )
    }

    /// Build the map; `None` when it has nothing to act on (Schwarz with no pair to act on).
    ///
    /// PROTOTYPE(#607): `below` is the map of the highest built rung below this one; the coarse
    /// rung reuses its Schwarz, or builds its own when there is none (the fixed config). A
    /// declined coarse space leaves the climb on the rung below (`None`) and the fixed config on
    /// its one-level Schwarz.
    fn build(
        &self,
        gramian: Gramian<'_>,
        below: Option<&Map>,
    ) -> Result<(Option<Map>, Vec<BuildWarning>), BuildError> {
        Ok(match self {
            Self::Diagonal => (Some(Map::Diagonal(diagonal_map(gramian)?)), Vec::new()),
            Self::Schwarz(schwarz) => {
                let (schwarz, warnings) = schwarz::build(gramian, &schwarz.local_solver)?;
                (schwarz.map(Map::Schwarz), warnings)
            }
            Self::Coarse(schwarz) => {
                let (one_level, mut warnings, reused) = match below {
                    Some(Map::Schwarz(one_level)) => (Some(one_level.clone()), Vec::new(), true),
                    _ => {
                        let (one_level, warnings) = schwarz::build(gramian, &schwarz.local_solver)?;
                        (one_level, warnings, false)
                    }
                };
                let Some(one_level) = one_level else {
                    return Ok((None, warnings));
                };
                match two_level::build(gramian, one_level.clone()) {
                    Ok(map) => (Some(Map::TwoLevel(map)), warnings),
                    Err(declined) => {
                        warnings.push(declined);
                        ((!reused).then_some(Map::Schwarz(one_level)), warnings)
                    }
                }
            }
        })
    }
}

impl Map {
    fn operator(&self) -> &dyn Operator {
        match self {
            Self::Diagonal(p) => p,
            Self::Schwarz(p) => p.operator(),
            Self::TwoLevel(p) => p,
        }
    }
}

#[derive(Deserialize)]
struct PreconditionerWire {
    config: PreconditionerConfig,
    rung: usize,
    map: Map,
    build_duration: Duration,
}

impl TryFrom<PreconditionerWire> for Preconditioner {
    type Error = String;

    fn try_from(wire: PreconditionerWire) -> Result<Self, String> {
        let PreconditionerWire {
            config,
            rung,
            map,
            build_duration,
        } = wire;
        if !config
            .rungs()
            .nth(rung)
            .is_some_and(|(r, _)| r.builds(&map))
        {
            return Err(format!("the map is not rung {rung} of {config:?}"));
        }
        Ok(Self {
            config,
            rung,
            map,
            build_duration,
        })
    }
}

impl Preconditioner {
    /// Stable display name of the map: `"Diagonal"` or `"Additive"`.
    pub fn variant_name(&self) -> &'static str {
        match self.map {
            Map::Diagonal(_) => "Diagonal",
            Map::Schwarz(_) => "Additive",
            Map::TwoLevel(_) => "TwoLevel",
        }
    }

    /// The configuration whose ladder this map is a rung of.
    pub fn config(&self) -> PreconditionerConfig {
        self.config.clone()
    }

    // `within` does not re-export `schwarz_precond::Operator`, so these are the only public way.
    /// Number of rows of the underlying linear operator.
    pub fn nrows(&self) -> usize {
        <Self as schwarz_precond::Operator>::nrows(self)
    }

    /// Number of columns of the underlying linear operator.
    pub fn ncols(&self) -> usize {
        <Self as schwarz_precond::Operator>::ncols(self)
    }

    /// Apply the preconditioner: writes `M^{-1} x` into `y`, on a stack allocated for this call.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        <Self as Operator>::apply(self, x, y)
    }

    /// What [`apply_with`](Self::apply_with) takes from its stack in the current rayon pool.
    pub fn scratch(&self) -> StackReq {
        <Self as Operator>::scratch(self)
    }

    /// [`apply`](Self::apply) on `stack`, which holds at least [`scratch`](Self::scratch), as
    /// [`MemBuffer::new`](crate::MemBuffer::new) gives; a caller can keep it across applies.
    pub fn apply_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        <Self as Operator>::apply_with(self, x, y, stack)
    }

    /// Time spent building this preconditioner.
    pub fn build_duration(&self) -> Duration {
        self.build_duration
    }
}

impl Operator for Preconditioner {
    fn nrows(&self) -> usize {
        self.map.operator().nrows()
    }

    fn ncols(&self) -> usize {
        self.map.operator().ncols()
    }

    fn scratch(&self) -> StackReq {
        self.map.operator().scratch()
    }

    fn apply_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        self.map.operator().apply_with(x, y, stack)
    }

    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        self.apply_with(x, y, stack)
    }
}

fn diagonal_map(gramian: Gramian<'_>) -> Result<DiagonalPreconditioner, BuildError> {
    let mut diag = gramian.diagonal();

    // A zero diagonal is an unidentified DOF, so the pseudo-inverse keeps it in the null space.
    for (index, d) in diag.iter_mut().enumerate() {
        if *d == 0.0 {
            continue;
        }
        let inv = 1.0 / *d;
        if !inv.is_finite() {
            return Err(BuildError::SingularDiagonal { index });
        }
        *d = inv;
    }

    Ok(DiagonalPreconditioner {
        inv_diag: Arc::from(diag),
    })
}
