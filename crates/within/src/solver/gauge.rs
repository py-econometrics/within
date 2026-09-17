//! Cross-term gauge directions (#297): a covariate another term reproduces to roundoff is a null
//! of the design, and nulls leave the solve space rather than wait for the preconditioner.

use std::sync::Mutex;

use schwarz_precond::Operator;

use crate::domain::collinearity::CollinearSlope;
use crate::domain::PreparedDesign;
use crate::linalg::dot;
use crate::operator::schwarz::Preconditioner;
use crate::operator::DesignOperator;
use crate::{AliasVerdict, BuildWarning};

/// Residual share at or below which a screened covariate is a null, not data: an exact alias
/// cancels to roundoff, a direction the data can still resolve sits orders above.
const GAUGE_NULL_TOL: f64 = 1e-20;

/// Share of a unit-norm proposal that must survive the rows already taken to become one.
/// Below it the proposal is a rescaled duplicate, and normalizing would amplify its roundoff.
const RANK_SHARE_TOL: f64 = 1e-6;

/// Cross-term null directions, orthonormal; `k × n_dofs`, row-major.
pub(crate) struct GaugeConstraint {
    rows: Vec<f64>,
    n_dofs: usize,
}

impl GaugeConstraint {
    /// Take the screen's null proposals out of the solve space, reporting each verdict.
    pub(crate) fn build(
        prepared: &PreparedDesign<'_>,
        screened: &[CollinearSlope],
    ) -> (Option<Self>, Vec<BuildWarning>) {
        let is_null = |slope: &CollinearSlope| slope.relative_residual <= GAUGE_NULL_TOL;
        let warnings = screened
            .iter()
            .map(|slope| {
                slope.warn(match is_null(slope) {
                    true => AliasVerdict::Constrained,
                    false => AliasVerdict::Kept,
                })
            })
            .collect();
        if !screened.iter().any(is_null) {
            return (None, warnings);
        }
        let n_dofs = prepared.design.n_dofs;
        let operator = DesignOperator::new(prepared);
        // Whitening leaves a level's columns orthogonal, so `Aᵀc ./ diag(AᵀA)` is its per-level fit.
        let scale = operator.column_norms_squared();
        let proposed = screened
            .iter()
            .filter(|slope| is_null(slope))
            .map(|slope| propose(prepared, &operator, &scale, slope))
            .collect();
        let gauge = Self {
            rows: orthonormalize(proposed, n_dofs),
            n_dofs,
        };
        match gauge.rank() {
            0 => (None, warnings),
            _ => (Some(gauge), warnings),
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.rows.len() / self.n_dofs
    }

    /// `x ← (I − VᵀV) x`.
    pub(crate) fn project(&self, x: &mut [f64]) {
        for row in self.rows.chunks_exact(self.n_dofs) {
            let share = dot(row, x);
            for (xi, &ri) in x.iter_mut().zip(row) {
                *xi -= share * ri;
            }
        }
    }
}

/// The screened covariate as a unit direction in coefficient space: its per-level fit inside the
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
    let norm = dot(&values, &values).sqrt();
    for value in &mut values {
        *value /= norm;
    }
    values
}

/// Pivoted Gram-Schmidt, so a near-duplicate is spent against the row it duplicates, never rescaled.
fn orthonormalize(mut directions: Vec<Vec<f64>>, n_dofs: usize) -> Vec<f64> {
    let mut basis: Vec<f64> = Vec::with_capacity(directions.len() * n_dofs);
    while let Some((next, share)) = directions
        .iter()
        .map(|d| dot(d, d).sqrt())
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
    {
        if share.is_nan() || share <= RANK_SHARE_TOL {
            break;
        }
        let mut row = directions.swap_remove(next);
        for value in &mut row {
            *value /= share;
        }
        for other in &mut directions {
            let overlap = dot(&row, other);
            for (o, &r) in other.iter_mut().zip(&row) {
                *o -= overlap * r;
            }
        }
        basis.extend_from_slice(&row);
    }
    basis
}

/// The base preconditioner restricted to the constrained solve space: `P M⁻¹ P`.
pub(crate) struct ConstrainedPreconditioner<'a> {
    base: &'a Preconditioner,
    gauge: &'a GaugeConstraint,
    /// The projected input; the preconditioner needs distinct in and out buffers.
    scratch: Mutex<Vec<f64>>,
}

impl<'a> ConstrainedPreconditioner<'a> {
    pub(crate) fn new(base: &'a Preconditioner, gauge: &'a GaugeConstraint) -> Self {
        Self {
            base,
            gauge,
            scratch: Mutex::new(vec![0.0; base.ncols()]),
        }
    }
}

impl Operator for ConstrainedPreconditioner<'_> {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }

    fn ncols(&self) -> usize {
        self.base.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let mut projected = self.scratch.lock().expect("uncontended scratch lock");
        projected.copy_from_slice(x);
        self.gauge.project(&mut projected);
        self.base.apply(&projected, y)?;
        self.gauge.project(y);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let mut projected = self.scratch.lock().expect("uncontended scratch lock");
        projected.copy_from_slice(x);
        self.gauge.project(&mut projected);
        <Preconditioner as Operator>::apply_adjoint(self.base, &projected, y)?;
        self.gauge.project(y);
        Ok(())
    }
}
