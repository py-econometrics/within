//! `M ≈ G⁻¹`: the diagonal and Schwarz preconditioners, read from `G`, and the opaque public
//! [`Preconditioner`] handle.

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

pub(crate) use ladder::{Current, Ladder};

/// Diagonal/Jacobi preconditioner for the fixed-effects Gramian.
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
}

impl Map {
    fn is_built_by(&self, rung: &Rung<'_>) -> bool {
        matches!(
            (self, rung),
            (Self::Diagonal(_), Rung::Diagonal) | (Self::Schwarz(_), Rung::Schwarz(_))
        )
    }

    fn operator(&self) -> &dyn Operator {
        match self {
            Self::Diagonal(p) => p,
            Self::Schwarz(p) => p.operator(),
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
            .is_some_and(|r| map.is_built_by(&r))
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

/// Build one rung's map; `None` when it has nothing to act on (Schwarz without a channel pair).
fn build_map(
    gramian: Gramian<'_>,
    rung: Rung<'_>,
) -> Result<(Option<Map>, Vec<BuildWarning>), BuildError> {
    Ok(match rung {
        Rung::Diagonal => (Some(Map::Diagonal(diagonal_map(gramian)?)), Vec::new()),
        Rung::Schwarz(schwarz) => {
            let (schwarz, warnings) = schwarz::build(gramian, &schwarz.local_solver)?;
            (schwarz.map(Map::Schwarz), warnings)
        }
    })
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
