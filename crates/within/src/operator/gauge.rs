//! Cross-term gauge directions (#297): a covariate another term reproduces to roundoff is a null
//! of the design, and nulls leave the solve space rather than wait for the preconditioner.

use schwarz_precond::{Operator, SolveError};

use crate::channel::Channel;
use crate::domain::collinearity::{CollinearSlope, GAUGE_NULL_TOL};
use crate::domain::PreparedDesign;
use crate::linalg::{dot, GramBasisWorkspace, RANK_TOL};
use crate::operator::DesignOperator;
use crate::AliasVerdict;

/// Cross-term null directions, orthonormal; `k × n_dofs`, row-major.
pub(crate) struct GaugeConstraint {
    rows: Vec<f64>,
    n_dofs: usize,
}

impl GaugeConstraint {
    /// The screen's null verdicts as one constraint, `None` when no independent direction survives.
    pub(crate) fn build(
        prepared: &PreparedDesign<'_>,
        screened: &[CollinearSlope],
    ) -> Option<Self> {
        let nulls: Vec<&CollinearSlope> = screened
            .iter()
            .filter(|slope| slope.verdict() == AliasVerdict::Constrained)
            .collect();
        if nulls.is_empty() {
            return None;
        }
        let n_dofs = prepared.design.n_dofs;
        let operator = DesignOperator::new(prepared);
        // Whitening leaves a level's columns orthogonal, so `Aᵀc ./ diag(AᵀA)` is its per-level fit.
        let scale = operator.column_norms_squared();
        let proposed: Vec<Vec<f64>> = nulls
            .iter()
            .map(|slope| propose(prepared, &operator, &scale, slope.slope, slope.term))
            .collect();
        // A contrast of near-parallel proposals divides their certified energy by its residual
        // share, so a share under certificate/tolerance is not itself a certified null.
        let certificate = nulls.iter().map(|s| s.certificate()).fold(0.0, f64::max);
        let k = proposed.len();
        let mut workspace =
            GramBasisWorkspace::new(k, (certificate / GAUGE_NULL_TOL).max(RANK_TOL));
        let w = workspace
            .orthonormalize(|gram| {
                for (j, a) in proposed.iter().enumerate() {
                    for (i, b) in proposed.iter().enumerate() {
                        gram[j * k + i] = dot(a, b);
                    }
                }
            })
            .rows;
        let mut rows = vec![0.0; w.len() / k * n_dofs];
        for (row, coefficients) in rows.chunks_exact_mut(n_dofs).zip(w.chunks_exact(k)) {
            for (&c, p) in coefficients.iter().zip(&proposed) {
                for (r, &pi) in row.iter_mut().zip(p) {
                    *r += c * pi;
                }
            }
        }
        let gauge = Self { rows, n_dofs };
        (gauge.rank() > 0).then_some(gauge)
    }

    pub(crate) fn rank(&self) -> usize {
        self.rows.len() / self.n_dofs
    }

    /// `y ← P M⁻¹ P x`, with `m` the base apply; both sides keep the operator self-adjoint.
    pub(crate) fn constrain(
        &self,
        x: &[f64],
        y: &mut [f64],
        m: impl FnOnce(&[f64], &mut [f64]) -> Result<(), SolveError>,
    ) -> Result<(), SolveError> {
        if x.len() != self.n_dofs || y.len() != self.n_dofs {
            return m(x, y);
        }
        // One copy per apply: a shared scratch needs a lock, and a batch applies from rayon workers
        // whose join can steal another apply that waits on it.
        let mut projected = x.to_vec();
        self.project(&mut projected);
        m(&projected, y)?;
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
    slope: Channel,
    term: usize,
) -> Vec<f64> {
    let design = &prepared.design;
    let c = design
        .raw_slope(slope)
        .expect("a screened slope carries a covariate");
    // Two intercepts alias through the ordinary FE gauge, not through the covariate.
    let (mut sum, mut total) = (0.0, 0.0);
    for (obs, &ci) in c.iter().enumerate() {
        let w = prepared.row_weight(obs);
        sum += w * ci;
        total += w;
    }
    let centered = total > 0.0
        && [slope.term, term]
            .iter()
            .all(|&t| design.terms[t].intercept);
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
    for (term, sign) in [(slope.term, 1.0), (term, -1.0)] {
        let t = &design.terms[term];
        let block = t.dofs();
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
