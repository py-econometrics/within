//! Cross-term gauge directions (#297): a covariate another term reproduces to roundoff is a null
//! of the design, and nulls leave the solve space rather than wait for the preconditioner.

use rayon::prelude::*;
use schwarz_precond::{Operator, SolveError};

use crate::channel::Channel;
use crate::domain::collinearity::{CollinearSlope, GAUGE_NULL_TOL};
use crate::domain::PreparedDesign;
use crate::linalg::{dot, GramBasisWorkspace, RANK_TOL};
use crate::operator::DesignOperator;
use crate::AliasVerdict;

/// Cross-term null directions, orthonormal; `k × n_dofs`, row-major.
#[derive(Clone)]
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
        let proposed: Vec<Vec<f64>> = nulls
            .iter()
            .map(|slope| propose(prepared, &operator, slope.slope, slope.term))
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

    /// This gauge against `map`: `W = M⁻¹ Vᵀ`, one base apply per direction.
    pub(crate) fn fold(self, map: &dyn Operator) -> GaugeFold {
        let n = self.n_dofs;
        let mut applied = vec![0.0; self.rows.len()];
        for (w, v) in applied.chunks_exact_mut(n).zip(self.rows.chunks_exact(n)) {
            map.apply(v, w)
                .expect("a built map applies at the dimension the solver checked");
        }
        GaugeFold {
            basis: self,
            applied,
        }
    }
}

/// Rows per reduction task; fixed, so the summation order ignores the thread count.
const CHUNK: usize = 4096;

/// A gauge against the map it sits beside: `V` and `W = M⁻¹ Vᵀ` by basis row, `k × n_dofs`.
pub(crate) struct GaugeFold {
    basis: GaugeConstraint,
    applied: Vec<f64>,
}

impl GaugeFold {
    /// The same directions against `map` instead: `W` is refolded, never carried over.
    pub(crate) fn refold(&self, map: &dyn Operator) -> GaugeFold {
        self.basis.clone().fold(map)
    }

    /// `y ← P M⁻¹ P x = P (M⁻¹ x − W V x)`, with `V x` reduced while the base apply runs.
    pub(crate) fn apply(
        &self,
        map: &dyn Operator,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), SolveError> {
        let n = self.basis.n_dofs;
        if x.len() != n || y.len() != n {
            return map.apply(x, y);
        }
        let (applied, shares) = rayon::join(|| map.apply(x, y), || self.shares(x));
        applied?;
        let k = self.basis.rank();
        let mut partials = vec![0.0; n.div_ceil(CHUNK) * k];
        partials
            .par_chunks_mut(k)
            .zip(y.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(i, (out, ys))| {
                let span = i * CHUNK..i * CHUNK + ys.len();
                for (&share, w) in shares.iter().zip(self.applied.chunks_exact(n)) {
                    for (yi, &wi) in ys.iter_mut().zip(&w[span.clone()]) {
                        *yi -= share * wi;
                    }
                }
                for (o, v) in out.iter_mut().zip(self.basis.rows.chunks_exact(n)) {
                    *o = dot(&v[span.clone()], ys);
                }
            });
        // The output projection stays explicit: `W V x` cancels `M⁻¹`'s amplified gauge share.
        let shares = sum_chunks(&partials, k);
        y.par_chunks_mut(CHUNK).enumerate().for_each(|(i, ys)| {
            let span = i * CHUNK..i * CHUNK + ys.len();
            for (&share, v) in shares.iter().zip(self.basis.rows.chunks_exact(n)) {
                for (yi, &vi) in ys.iter_mut().zip(&v[span.clone()]) {
                    *yi -= share * vi;
                }
            }
        });
        Ok(())
    }

    /// `V x`, summed chunk by chunk in index order.
    fn shares(&self, x: &[f64]) -> Vec<f64> {
        let (k, n) = (self.basis.rank(), self.basis.n_dofs);
        let mut partials = vec![0.0; n.div_ceil(CHUNK) * k];
        partials
            .par_chunks_mut(k)
            .zip(x.par_chunks(CHUNK))
            .enumerate()
            .for_each(|(i, (out, xs))| {
                let span = i * CHUNK..i * CHUNK + xs.len();
                for (o, v) in out.iter_mut().zip(self.basis.rows.chunks_exact(n)) {
                    *o = dot(&v[span.clone()], xs);
                }
            });
        sum_chunks(&partials, k)
    }
}

/// Per-chunk partial sums of width `k`, added in chunk order.
fn sum_chunks(partials: &[f64], k: usize) -> Vec<f64> {
    let mut sums = vec![0.0; k];
    for part in partials.chunks_exact(k) {
        for (s, &p) in sums.iter_mut().zip(part) {
            *s += p;
        }
    }
    sums
}

impl std::fmt::Debug for GaugeFold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaugeFold")
            .field("rank", &self.basis.rank())
            .field("n_dofs", &self.basis.n_dofs)
            .finish()
    }
}

/// The screened covariate as a direction in coefficient space: its per-level fit inside the
/// carrying term against the same fit inside the term that reproduces it.
fn propose(
    prepared: &PreparedDesign<'_>,
    operator: &DesignOperator<'_>,
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
    // Whitening leaves a level's columns orthogonal, so `Aᵀc ./ diag(AᵀA)` is its per-level fit.
    for (block_term, sign) in [(slope.term, 1.0), (term, -1.0)] {
        let block = design.terms[block_term].dofs();
        for ((v, &f), &s) in values[block.clone()]
            .iter_mut()
            .zip(&fit[block])
            .zip(prepared.diagonal(block_term))
        {
            *v = match s > 0.0 {
                true => sign * f / s,
                false => 0.0,
            };
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `M⁻¹ = D + u uᵀ`: symmetric, and coupling every coordinate, so no chunk is self-contained.
    struct DiagonalPlusRankOne {
        diagonal: Vec<f64>,
        u: Vec<f64>,
    }

    impl Operator for DiagonalPlusRankOne {
        fn nrows(&self) -> usize {
            self.diagonal.len()
        }

        fn ncols(&self) -> usize {
            self.diagonal.len()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            let share = dot(&self.u, x);
            for (((yi, &xi), &di), &ui) in y.iter_mut().zip(x).zip(&self.diagonal).zip(&self.u) {
                *yi = di * xi + share * ui;
            }
            Ok(())
        }

        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
            self.apply(x, y)
        }
    }

    impl GaugeFold {
        /// Number of directions folded against the map.
        pub(crate) fn rank_for_test(&self) -> usize {
            self.basis.rank()
        }
    }

    /// `x ← (I − VᵀV) x`, one direction at a time.
    fn project(basis: &GaugeConstraint, x: &mut [f64]) {
        for row in basis.rows.chunks_exact(basis.n_dofs) {
            let share = dot(row, x);
            for (xi, &ri) in x.iter_mut().zip(row) {
                *xi -= share * ri;
            }
        }
    }

    #[test]
    fn the_folded_apply_is_the_two_sided_projection() {
        let n = 3 * CHUNK + 17;
        let wave = |f: f64| (0..n).map(|i| (f * i as f64).sin()).collect::<Vec<_>>();
        let first = wave(0.37);
        let norm = dot(&first, &first).sqrt();
        let first: Vec<f64> = first.iter().map(|v| v / norm).collect();
        let mut second = wave(1.13);
        let overlap = dot(&first, &second);
        second
            .iter_mut()
            .zip(&first)
            .for_each(|(s, &f)| *s -= overlap * f);
        let norm = dot(&second, &second).sqrt();
        second.iter_mut().for_each(|s| *s /= norm);
        let basis = GaugeConstraint {
            rows: [first, second].concat(),
            n_dofs: n,
        };
        let m = DiagonalPlusRankOne {
            diagonal: (0..n).map(|i| 1.0 + (i % 7) as f64).collect(),
            u: wave(0.05),
        };
        let x = wave(2.71);

        let mut projected = x.clone();
        project(&basis, &mut projected);
        let mut reference = vec![0.0; n];
        m.apply(&projected, &mut reference).expect("apply");
        project(&basis, &mut reference);

        let fold = basis.fold(&m);
        let mut y = vec![0.0; n];
        fold.apply(&m, &x, &mut y).expect("apply");
        let scale = reference.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let error = y
            .iter()
            .zip(&reference)
            .fold(0.0f64, |a, (y, r)| a.max((y - r).abs()));
        assert!(
            error <= 1e-12 * scale,
            "error {error:.3e} at scale {scale:.3e}"
        );
    }
}
