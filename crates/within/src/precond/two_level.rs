//! PROTOTYPE(#607): the two-level map `PᵀBP + Q`: one-level Schwarz `B` balanced by the coarse
//! solve `Q = Z E⁻¹ Zᵀ`, with `E = ZᵀSZ` and `P = I − SQ`. Not persisted.

use std::sync::Arc;
use std::time::Instant;

use schwarz_precond::{MemStack, Operator, SolveError, StackReq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::schwarz::FeSchwarz;
use crate::config::probe;
use crate::system::{CoarseSolve, Gramian};
use crate::BuildWarning;

/// `PᵀBP + Q` with `P = I − SQ`: exact on the coarse space, `M S Z = Z`.
#[derive(Clone, Debug)]
pub(super) struct TwoLevelMap {
    one_level: FeSchwarz,
    coarse: Arc<CoarseSolve>,
}

/// `one_level` with the rigid coarse correction over `gramian`'s unknowns, or the warning saying
/// why there is none.
pub(super) fn build(
    gramian: Gramian<'_>,
    one_level: FeSchwarz,
) -> Result<TwoLevelMap, BuildWarning> {
    let started = Instant::now();
    match gramian.rigid_coarse() {
        Ok(coarse) => {
            if probe::verbose() {
                eprintln!(
                    "[probe-coarse] built columns={} hubs={} factor_values={} nnz={} secs={:.3}",
                    coarse.n_columns(),
                    coarse.n_hubs(),
                    coarse.factor_values(),
                    coarse.nnz(),
                    started.elapsed().as_secs_f64()
                );
            }
            Ok(TwoLevelMap {
                one_level,
                coarse: Arc::new(coarse),
            })
        }
        Err(decline) => {
            if probe::verbose() {
                eprintln!(
                    "[probe-coarse] declined {decline:?} secs={:.3}",
                    started.elapsed().as_secs_f64()
                );
            }
            Err(BuildWarning::CoarseDeclined {
                reason: format!("{decline:?}"),
            })
        }
    }
}

impl Serialize for TwoLevelMap {
    fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "the prototype's two-level map is not persisted",
        ))
    }
}

impl<'de> Deserialize<'de> for TwoLevelMap {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "the prototype's two-level map is not persisted",
        ))
    }
}

impl Operator for TwoLevelMap {
    fn nrows(&self) -> usize {
        self.one_level.operator().nrows()
    }

    fn ncols(&self) -> usize {
        self.one_level.operator().ncols()
    }

    fn scratch(&self) -> StackReq {
        self.one_level.operator().scratch()
    }

    /// `Z E⁻¹ Zᵀx + (I − Z E⁻¹ (SZ)ᵀ) B (I − SZ E⁻¹ Zᵀ) x`.
    fn apply_with(&self, x: &[f64], y: &mut [f64], stack: &mut MemStack) -> Result<(), SolveError> {
        let coarse = &self.coarse;
        let m = coarse.n_columns();
        let mut c = vec![0.0; m];
        coarse.z_t_mul(x, &mut c);
        coarse.solve(&mut c);
        let mut t = x.to_vec();
        coarse.sub_sz(&c, &mut t);
        self.one_level.operator().apply_with(&t, y, stack)?;
        let mut d = vec![0.0; m];
        coarse.sz_t_mul(y, &mut d);
        coarse.solve(&mut d);
        for (c, d) in c.iter_mut().zip(&d) {
            *c -= d;
        }
        coarse.add_z(&c, y);
        Ok(())
    }

    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.apply_with(x, y, stack)
    }
}

/// PROTOTYPE(#612): any map `inner` deflated by a coarse solve: `PᵀMP + Q`, as [`TwoLevelMap`].
pub(crate) struct Deflated<'a> {
    pub(crate) inner: &'a dyn Operator,
    pub(crate) coarse: &'a CoarseSolve,
}

impl Operator for Deflated<'_> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn scratch(&self) -> StackReq {
        self.inner.scratch()
    }

    fn apply_with(&self, x: &[f64], y: &mut [f64], stack: &mut MemStack) -> Result<(), SolveError> {
        let coarse = self.coarse;
        let m = coarse.n_columns();
        let mut c = vec![0.0; m];
        coarse.z_t_mul(x, &mut c);
        coarse.solve(&mut c);
        let mut t = x.to_vec();
        coarse.sub_sz(&c, &mut t);
        self.inner.apply_with(&t, y, stack)?;
        let mut d = vec![0.0; m];
        coarse.sz_t_mul(y, &mut d);
        coarse.solve(&mut d);
        for (c, d) in c.iter_mut().zip(&d) {
            *c -= d;
        }
        coarse.add_z(&c, y);
        Ok(())
    }

    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.apply_with(x, y, stack)
    }
}
