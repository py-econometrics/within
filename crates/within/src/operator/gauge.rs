//! Gauge constraint (#297): the collinearity screen proposes directions, a backward-error test
//! on the orthonormal basis decides which are null, and those leave the solve space.

use std::sync::Mutex;

use schwarz_precond::Operator;

use crate::domain::gauge::GaugeCandidate;
use crate::domain::level_moments::TermMoments;
use crate::domain::Design;
use crate::linalg::dot;
use crate::operator::schwarz::Preconditioner;
use crate::operator::DesignOperator;
use crate::BuildWarning;

/// Share of a unit-norm candidate that must survive the rows already taken to become one.
/// Below it the candidate is a rescaled duplicate, and normalizing would amplify its image.
const RANK_SHARE_TOL: f64 = 1e-6;

/// `‖A n‖ ≤ NULL_TOL · ‖ |A| |n| ‖` certifies a row as null: a backward error, so a uniform
/// weight change cannot move the verdict. An alias lands at roundoff, a real direction far above.
const NULL_TOL: f64 = 1e-11;

/// Certified gauge directions, orthonormal; `rows` is `k × n_dofs`, row-major.
pub(crate) struct GaugeConstraint {
    n_dofs: usize,
    rows: Vec<f64>,
}

impl GaugeConstraint {
    /// `None` when no proposed direction survives the certificate.
    pub(crate) fn new(
        candidates: Vec<GaugeCandidate>,
        design: &Design<'_>,
        moments: &TermMoments,
        sqrt_weights: Option<&[f64]>,
        warnings: &[BuildWarning],
    ) -> Option<Self> {
        let n_dofs = design.n_dofs;
        let design_op = DesignOperator::new(design, sqrt_weights);
        let mut dof_scratch = vec![0.0f64; n_dofs];
        let mut directions: Vec<Vec<f64>> = Vec::new();
        for candidate in candidates {
            let BuildWarning::CollinearSlopeCovariate { slope, term, .. } =
                warnings[candidate.warning]
            else {
                continue;
            };
            // Whitening leaves a level's columns orthonormal, so `Dᵀ W c` IS its per-level fit.
            design_op
                .apply_adjoint(&candidate.weighted, &mut dof_scratch)
                .expect("the design operator cannot fail");
            let mut values = vec![0.0f64; n_dofs];
            for (term, sign) in [(slope.term, 1.0), (term, -1.0)] {
                let meta = &design.terms[term];
                let block = meta.offset..meta.offset + meta.n_dofs();
                for (v, &fit) in values[block.clone()].iter_mut().zip(&dof_scratch[block]) {
                    *v = sign * fit;
                }
                // The constant's channel holds a weighted sum, one division short of a fit.
                let Some(column) = meta.intercept_column() else {
                    continue;
                };
                for level in 0..meta.n_levels() {
                    let w = moments[term].w_sum(level);
                    let slot = &mut values[meta.column_base(column) + level];
                    *slot = match w > 0.0 {
                        true => *slot / w,
                        false => 0.0,
                    };
                }
            }
            let norm = dot(&values, &values).sqrt();
            if norm <= 0.0 || !norm.is_finite() {
                continue;
            }
            for value in &mut values {
                *value /= norm;
            }
            directions.push(values);
        }

        // Pivoted, so a near-duplicate is spent against the row it duplicates, never rescaled.
        let mut rows: Vec<f64> = Vec::new();
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
                // Twice, because one pass loses orthogonality exactly where the shares are small.
                for _ in 0..2 {
                    let overlap = dot(&row, other);
                    for (o, &r) in other.iter_mut().zip(&row) {
                        *o -= overlap * r;
                    }
                }
            }
            rows.extend_from_slice(&row);
        }
        let k = rows.len() / n_dofs;
        if k == 0 {
            return None;
        }

        // Per row against the share of the Frobenius budget it may spend, so `‖A N‖_F` holds.
        let budget = NULL_TOL / (k as f64).sqrt();
        let scale = design_op.column_norms_squared();
        let mut obs = vec![0.0f64; design_op.nrows()];
        let mut certified: Vec<f64> = Vec::with_capacity(rows.len());
        for row in rows.chunks_exact(n_dofs) {
            design_op.apply(row, &mut obs).ok()?;
            let reference: f64 = row.iter().zip(&scale).map(|(&r, &s)| r * r * s).sum();
            if dot(&obs, &obs).sqrt() <= budget * reference.sqrt() {
                certified.extend_from_slice(row);
            }
        }
        (!certified.is_empty()).then_some(Self {
            n_dofs,
            rows: certified,
        })
    }

    /// `x ← (I − VVᵀ) x`.
    fn project(&self, x: &mut [f64]) {
        for row in self.rows.chunks_exact(self.n_dofs) {
            let share = dot(row, x);
            for (xi, &ri) in x.iter_mut().zip(row) {
                *xi -= share * ri;
            }
        }
    }
}

/// The base preconditioner restricted to the constrained solve space: `P M⁻¹ P`.
pub(crate) struct GaugePreconditioner<'a> {
    base: &'a Preconditioner,
    constraint: &'a GaugeConstraint,
    /// The projected input; the preconditioner needs distinct in and out buffers.
    scratch: Mutex<Vec<f64>>,
}

impl<'a> GaugePreconditioner<'a> {
    pub(crate) fn new(base: &'a Preconditioner, constraint: &'a GaugeConstraint) -> Self {
        Self {
            base,
            constraint,
            scratch: Mutex::new(vec![0.0; base.ncols()]),
        }
    }
}

impl Operator for GaugePreconditioner<'_> {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }

    fn ncols(&self) -> usize {
        self.base.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let mut projected = self.scratch.lock().unwrap();
        projected.copy_from_slice(x);
        self.constraint.project(&mut projected);
        self.base.apply(&projected, y)?;
        self.constraint.project(y);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let mut projected = self.scratch.lock().unwrap();
        projected.copy_from_slice(x);
        self.constraint.project(&mut projected);
        <Preconditioner as Operator>::apply_adjoint(self.base, &projected, y)?;
        self.constraint.project(y);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl GaugeConstraint {
        pub(crate) fn rank_for_test(&self) -> usize {
            self.rows.len() / self.n_dofs
        }
    }
}
