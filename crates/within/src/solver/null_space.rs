//! The solve space's known null directions (#297): whitening's dropped columns, and cross-term
//! aliasing the collinearity screen proposes and a backward-error test certifies.

use std::sync::Mutex;

use schwarz_precond::Operator;

use super::CoefficientPosition;
use crate::domain::Design;
use crate::linalg::dot;
use crate::operator::schwarz::Preconditioner;
use crate::operator::DesignOperator;
use crate::{AliasVerdict, BuildWarning};

/// Share of a unit-norm candidate that must survive the rows already taken to become one.
/// Below it the candidate is a rescaled duplicate, and normalizing would amplify its image.
const RANK_SHARE_TOL: f64 = 1e-6;

/// `‖A n‖ ≤ NULL_TOL · ‖ |A| |n| ‖` certifies a row as null: a backward error, so a uniform
/// weight change cannot move the verdict. An alias lands at roundoff, a real direction far above.
const NULL_TOL: f64 = 1e-11;

/// Every direction the solve excludes; the operator, the preconditioner and the result read this.
pub(crate) struct NullSpace {
    /// Exact-zero whitened columns, ascending in `(term, level, column)`.
    pub(super) dropped: Vec<CoefficientPosition>,
    /// Certified cross-term directions, orthonormal; `k × n_dofs`, row-major.
    pub(super) rows: Vec<f64>,
    pub(super) n_dofs: usize,
}

/// One warned covariate, centered so the two per-level fits differ by the aliasing alone and
/// weighted so the design's adjoint reads it directly.
pub(super) struct AliasCandidate {
    /// Index of the warning this came from, which names both terms and takes the verdict.
    warning: usize,
    weighted: Vec<f64>,
}

impl AliasCandidate {
    /// Must run before whitening, which overwrites the covariate's column.
    pub(super) fn capture(
        design: &Design<'_>,
        weights: Option<&[f64]>,
        warnings: &[BuildWarning],
    ) -> Vec<Self> {
        warnings
            .iter()
            .enumerate()
            .filter_map(|(warning, w)| {
                let BuildWarning::CollinearSlopeCovariate { slope, term, .. } = w else {
                    return None;
                };
                let covariate = *design.loading(*slope).covariate()?;
                let c = design.loading_column(covariate as usize);
                let (mut sum, mut total) = (0.0, 0.0);
                for (obs, &ci) in c.iter().enumerate() {
                    let w = weights.map_or(1.0, |ws| ws[obs]);
                    sum += w * ci;
                    total += w;
                }
                // Two intercepts alias through the ordinary FE gauge, not the covariate.
                let intercepts = [slope.term, *term]
                    .iter()
                    .all(|&t| design.terms[t].intercept_column().is_some());
                let origin = match intercepts && total > 0.0 {
                    true => sum / total,
                    false => 0.0,
                };
                Some(Self {
                    warning,
                    weighted: c
                        .iter()
                        .enumerate()
                        .map(|(obs, &ci)| (ci - origin) * weights.map_or(1.0, |ws| ws[obs].sqrt()))
                        .collect(),
                })
            })
            .collect()
    }
}

impl NullSpace {
    pub(crate) fn rank(&self) -> usize {
        self.rows.len() / self.n_dofs
    }

    /// Certify the proposals against the whitened design and record each verdict on its warning.
    pub(super) fn constrain(
        &mut self,
        candidates: Vec<AliasCandidate>,
        design: &Design<'_>,
        weights: Option<&[f64]>,
        warnings: &mut [BuildWarning],
    ) {
        if candidates.is_empty() {
            return;
        }
        let n_dofs = self.n_dofs;
        let sqrt_weights: Option<Vec<f64>> =
            weights.map(|w| w.iter().map(|&wi| wi.sqrt()).collect());
        let design_op = DesignOperator::new(design, sqrt_weights.as_deref());
        // Whitening leaves a level's columns orthogonal, so `Aᵀc / diag(AᵀA)` is its per-level fit.
        let scale = design_op.column_norms_squared();
        let mut fit = vec![0.0f64; n_dofs];
        let mut proposed: Vec<(usize, Vec<f64>)> = Vec::new();
        for candidate in candidates {
            let BuildWarning::CollinearSlopeCovariate { slope, term, .. } =
                warnings[candidate.warning]
            else {
                continue;
            };
            design_op
                .apply_adjoint(&candidate.weighted, &mut fit)
                .expect("the design operator cannot fail");
            let mut values = vec![0.0f64; n_dofs];
            for (term, sign) in [(slope.term, 1.0), (term, -1.0)] {
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
            if norm <= 0.0 || !norm.is_finite() {
                continue;
            }
            for value in &mut values {
                *value /= norm;
            }
            proposed.push((candidate.warning, values));
        }

        // Pivoted, so a near-duplicate is spent against the row it duplicates, never rescaled.
        let mut directions: Vec<Vec<f64>> = proposed.iter().map(|(_, d)| d.clone()).collect();
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
            return;
        }

        // Per row against the share of the Frobenius budget it may spend, so `‖A N‖_F` holds.
        let budget = NULL_TOL / (k as f64).sqrt();
        let mut obs = vec![0.0f64; design_op.nrows()];
        for row in rows.chunks_exact(n_dofs) {
            design_op
                .apply(row, &mut obs)
                .expect("the design operator cannot fail");
            let reference: f64 = row.iter().zip(&scale).map(|(&r, &s)| r * r * s).sum();
            if dot(&obs, &obs).sqrt() <= budget * reference.sqrt() {
                self.rows.extend_from_slice(row);
            }
        }

        // A proposal is removed exactly when the certified rows span it.
        let mut residual = vec![0.0f64; n_dofs];
        for (warning, direction) in &proposed {
            residual.copy_from_slice(direction);
            self.project(&mut residual);
            let verdict = match dot(&residual, &residual).sqrt() <= RANK_SHARE_TOL {
                true => AliasVerdict::Constrained,
                false => AliasVerdict::Kept,
            };
            if let BuildWarning::CollinearSlopeCovariate { verdict: slot, .. } =
                &mut warnings[*warning]
            {
                *slot = verdict;
            }
        }
    }

    /// `x ← (I − VVᵀ) x`.
    pub(crate) fn project(&self, x: &mut [f64]) {
        for row in self.rows.chunks_exact(self.n_dofs) {
            let share = dot(row, x);
            for (xi, &ri) in x.iter_mut().zip(row) {
                *xi -= share * ri;
            }
        }
    }
}

/// The base preconditioner restricted to the constrained solve space: `P M⁻¹ P`.
pub(crate) struct ProjectedPreconditioner<'a> {
    base: &'a Preconditioner,
    null: &'a NullSpace,
    /// The projected input; the preconditioner needs distinct in and out buffers.
    scratch: Mutex<Vec<f64>>,
}

impl<'a> ProjectedPreconditioner<'a> {
    pub(crate) fn new(base: &'a Preconditioner, null: &'a NullSpace) -> Self {
        Self {
            base,
            null,
            scratch: Mutex::new(vec![0.0; base.ncols()]),
        }
    }
}

impl Operator for ProjectedPreconditioner<'_> {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }

    fn ncols(&self) -> usize {
        self.base.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let mut projected = self.scratch.lock().unwrap();
        projected.copy_from_slice(x);
        self.null.project(&mut projected);
        self.base.apply(&projected, y)?;
        self.null.project(y);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        let mut projected = self.scratch.lock().unwrap();
        projected.copy_from_slice(x);
        self.null.project(&mut projected);
        <Preconditioner as Operator>::apply_adjoint(self.base, &projected, y)?;
        self.null.project(y);
        Ok(())
    }
}
