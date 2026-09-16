//! Cross-term gauge directions (#297): the collinearity screen proposes them, a backward-error
//! test against the whitened design certifies them, and certified rows leave the solve space.

use std::sync::Mutex;

use schwarz_precond::Operator;

use crate::domain::collinearity::CollinearSlope;
use crate::domain::PreparedDesign;
use crate::linalg::dot;
use crate::operator::schwarz::Preconditioner;
use crate::operator::DesignOperator;
use crate::{AliasVerdict, BuildWarning};

/// Share of a unit-norm proposal that must survive the rows already taken to become one.
/// Below it the proposal is a rescaled duplicate, and normalizing would amplify its image.
const RANK_SHARE_TOL: f64 = 1e-6;

/// `‖A n‖ ≤ NULL_TOL · ‖ |A| |n| ‖` certifies a row as null: a backward error, so a uniform
/// weight change cannot move the verdict. An alias lands at roundoff, a real direction far above.
const NULL_TOL: f64 = 1e-11;

/// Certified cross-term null directions, orthonormal; `k × n_dofs`, row-major.
pub(crate) struct GaugeConstraint {
    rows: Vec<f64>,
    n_dofs: usize,
}

impl GaugeConstraint {
    /// Certify the screen's proposals against the whitened design, reporting each verdict.
    pub(crate) fn build(
        prepared: &PreparedDesign<'_>,
        screened: &[CollinearSlope],
    ) -> (Option<Self>, Vec<BuildWarning>) {
        if screened.is_empty() {
            return (None, Vec::new());
        }
        let n_dofs = prepared.design.n_dofs;
        let operator = DesignOperator::new(prepared);
        // Whitening leaves a level's columns orthogonal, so `Aᵀc ./ diag(AᵀA)` is its per-level fit.
        let scale = operator.column_norms_squared();
        let proposed: Vec<(usize, Vec<f64>)> = screened
            .iter()
            .enumerate()
            .filter_map(|(index, slope)| {
                Some((index, propose(prepared, &operator, &scale, slope)?))
            })
            .collect();

        let gauge = Self {
            rows: certify(&operator, &scale, &proposed, n_dofs),
            n_dofs,
        };

        // A proposal is constrained exactly when the certified rows span it.
        let mut verdicts = vec![AliasVerdict::Kept; screened.len()];
        let mut residual = vec![0.0f64; n_dofs];
        for (index, direction) in &proposed {
            residual.copy_from_slice(direction);
            gauge.project(&mut residual);
            if dot(&residual, &residual).sqrt() <= RANK_SHARE_TOL {
                verdicts[*index] = AliasVerdict::Constrained;
            }
        }
        let warnings = screened
            .iter()
            .zip(verdicts)
            .map(|(slope, verdict)| slope.warn(verdict))
            .collect();

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
) -> Option<Vec<f64>> {
    let design = &prepared.design;
    let covariate = *design.loading(screened.slope).covariate()?;
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
    (norm > 0.0 && norm.is_finite()).then(|| {
        for value in &mut values {
            *value /= norm;
        }
        values
    })
}

/// Orthonormalize the proposals, then keep the rows the design agrees carry nothing.
fn certify(
    operator: &DesignOperator<'_>,
    scale: &[f64],
    proposed: &[(usize, Vec<f64>)],
    n_dofs: usize,
) -> Vec<f64> {
    // Pivoted, so a near-duplicate is spent against the row it duplicates, never rescaled.
    let mut directions: Vec<Vec<f64>> = proposed.iter().map(|(_, d)| d.clone()).collect();
    let mut basis: Vec<f64> = Vec::new();
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

    // Per row against the share of the Frobenius budget it may spend, so `‖A N‖_F` holds.
    let budget = NULL_TOL / ((basis.len() / n_dofs).max(1) as f64).sqrt();
    let mut obs = vec![0.0f64; operator.nrows()];
    let mut rows = Vec::new();
    for row in basis.chunks_exact(n_dofs) {
        operator
            .apply(row, &mut obs)
            .expect("the design operator cannot fail");
        let reference: f64 = row.iter().zip(scale).map(|(&r, &s)| r * r * s).sum();
        if dot(&obs, &obs).sqrt() <= budget * reference.sqrt() {
            rows.extend_from_slice(row);
        }
    }
    rows
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
