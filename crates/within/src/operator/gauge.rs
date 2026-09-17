//! Cross-term gauge directions (#297): a covariate another term reproduces to roundoff is a null
//! of the design, and nulls leave the solve space rather than wait for the preconditioner.

use std::cell::RefCell;

use schwarz_precond::{Operator, SolveError};
use serde::{Deserialize, Serialize};

use crate::domain::collinearity::CollinearSlope;
use crate::domain::{BasisScratch, PreparedDesign, RANK_TOL};
use crate::linalg::dot;
use crate::operator::DesignOperator;
use crate::AliasVerdict;

/// Residual share at or below which a screened covariate is a null, not data: an exact alias
/// cancels to roundoff, a direction the data can still resolve sits orders above.
const GAUGE_NULL_TOL: f64 = 1e-20;

thread_local! {
    /// The projected input of a constrained apply; the base preconditioner needs distinct buffers.
    static PROJECTED: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// What the screen's residual says a warned direction is.
pub(crate) fn verdict(slope: &CollinearSlope) -> AliasVerdict {
    match slope.relative_residual <= GAUGE_NULL_TOL {
        true => AliasVerdict::Constrained,
        false => AliasVerdict::Kept,
    }
}

/// Cross-term null directions, orthonormal; `k × n_dofs`, row-major.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct GaugeConstraint {
    rows: Vec<f64>,
    n_dofs: usize,
}

impl GaugeConstraint {
    /// The screen's null proposals, `None` when no independent direction survives.
    pub(crate) fn build(
        prepared: &PreparedDesign<'_>,
        screened: &[CollinearSlope],
    ) -> Option<Self> {
        let nulls: Vec<&CollinearSlope> = screened
            .iter()
            .filter(|slope| verdict(slope) == AliasVerdict::Constrained)
            .collect();
        if nulls.is_empty() {
            return None;
        }
        let n_dofs = prepared.design.n_dofs;
        let operator = DesignOperator::new(prepared);
        // Whitening leaves a level's columns orthogonal, so `Aᵀc ./ diag(AᵀA)` is its per-level fit.
        let scale = operator.column_norms_squared();
        let proposed = nulls
            .iter()
            .map(|slope| propose(prepared, &operator, &scale, slope))
            .collect();
        let gauge = Self {
            rows: orthonormalize(proposed, n_dofs),
            n_dofs,
        };
        (gauge.rank() > 0).then_some(gauge)
    }

    pub(crate) fn rank(&self) -> usize {
        self.rows.len() / self.n_dofs
    }

    /// `y ← P M⁻¹ P x`, with `m` the base apply.
    pub(crate) fn constrain(
        &self,
        x: &[f64],
        y: &mut [f64],
        m: impl FnOnce(&[f64], &mut [f64]) -> Result<(), SolveError>,
    ) -> Result<(), SolveError> {
        PROJECTED.with_borrow_mut(|projected| {
            projected.clear();
            projected.extend_from_slice(x);
            self.project(projected);
            m(projected, y)
        })?;
        self.project(y);
        Ok(())
    }

    /// `x ← (I − VᵀV) x`.
    fn project(&self, x: &mut [f64]) {
        for row in self.rows.chunks_exact(self.n_dofs) {
            let share = dot(row, x);
            for (xi, &ri) in x.iter_mut().zip(row) {
                *xi -= share * ri;
            }
        }
    }
}

impl std::fmt::Debug for GaugeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaugeConstraint")
            .field("rank", &self.rank())
            .field("n_dofs", &self.n_dofs)
            .finish()
    }
}

/// The screened covariate as a direction in coefficient space: its per-level fit inside the
/// carrying term against the same fit inside the term that reproduces it.
fn propose(
    prepared: &PreparedDesign<'_>,
    operator: &DesignOperator<'_>,
    scale: &[f64],
    screened: &CollinearSlope,
) -> Vec<f64> {
    let design = &prepared.design;
    let covariate = *design
        .loading(screened.slope)
        .covariate()
        .expect("a screened slope carries a covariate");
    let c = design.frame.loading_column(covariate as usize);
    // Two intercepts alias through the ordinary FE gauge, not through the covariate.
    let (mut sum, mut total) = (0.0, 0.0);
    for (obs, &ci) in c.iter().enumerate() {
        let w = prepared.row_weight(obs);
        sum += w * ci;
        total += w;
    }
    let centered = total > 0.0
        && [screened.slope.term, screened.term]
            .iter()
            .all(|&t| design.terms[t].has_intercept());
    let origin = match centered {
        true => sum / total,
        false => 0.0,
    };
    let weighted: Vec<f64> = c
        .iter()
        .enumerate()
        .map(|(obs, &ci)| (ci - origin) * prepared.row_weight(obs).sqrt())
        .collect();

    let mut fit = vec![0.0f64; design.n_dofs];
    operator
        .apply_adjoint(&weighted, &mut fit)
        .expect("the design operator cannot fail");

    let mut values = vec![0.0f64; design.n_dofs];
    for (term, sign) in [(screened.slope.term, 1.0), (screened.term, -1.0)] {
        let meta = &design.terms[term];
        let block = meta.offset..meta.offset + meta.n_dofs();
        for ((v, &f), &s) in values[block.clone()]
            .iter_mut()
            .zip(&fit[block.clone()])
            .zip(&scale[block])
        {
            *v = match s > 0.0 {
                true => sign * f / s,
                false => 0.0,
            };
        }
    }
    values
}

/// Pivoted Gram-Schmidt in the proposals' Gram, by the rank rule a slope column already obeys.
fn orthonormalize(proposed: Vec<Vec<f64>>, n_dofs: usize) -> Vec<f64> {
    let k = proposed.len();
    let mut scratch = BasisScratch::new(k);
    for (j, a) in proposed.iter().enumerate() {
        for (i, b) in proposed.iter().enumerate() {
            scratch.gram[j * k + i] = dot(a, b);
        }
    }
    scratch.orthonormalize(k, RANK_TOL);
    let mut rows = vec![0.0; scratch.basis.len() / k * n_dofs];
    for (row, coefficients) in rows
        .chunks_exact_mut(n_dofs)
        .zip(scratch.basis.chunks_exact(k))
    {
        for (&c, p) in coefficients.iter().zip(&proposed) {
            for (r, &pi) in row.iter_mut().zip(p) {
                *r += c * pi;
            }
        }
    }
    rows
}
