//! Shared dense linear-algebra kernels.

/// Neumaier accumulation keeps long Gram diagonals and cross-cells consistent
/// without enlarging the matrix validation budget.
#[inline]
pub(crate) fn add_compensated(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    if sum.abs() >= value.abs() {
        *correction += (*sum - next) + value;
    } else {
        *correction += (value - next) + *sum;
    }
    *sum = next;
}

/// The columns of a basis over one block of rows whose Gram matrix is diagonal; callers size
/// the moment slices to its width.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
pub(crate) trait DiagonalGramBasis {
    /// Row `p`'s entry in column `c`.
    fn entry(&self, p: usize, c: usize) -> f64;
    /// Column `c`'s squared norm; a column with none fits nothing.
    fn norm2(&self, c: usize) -> f64;
}

/// Row-major `entries`, `norms.len()` per row, with their columns' squared norms.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
pub(crate) struct DenseBasis<'a> {
    pub(crate) entries: &'a [f64],
    pub(crate) norms: &'a [f64],
}

impl DiagonalGramBasis for DenseBasis<'_> {
    #[inline(always)]
    fn entry(&self, p: usize, c: usize) -> f64 {
        self.entries[p * self.norms.len() + c]
    }

    #[inline(always)]
    fn norm2(&self, c: usize) -> f64 {
        self.norms[c]
    }
}

/// Adds row `p`'s `entry · v` to each column's moment.
#[inline(always)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
fn add_moments(basis: &impl DiagonalGramBasis, p: usize, v: f64, moments: &mut [f64]) {
    for (c, m) in moments.iter_mut().enumerate() {
        *m += basis.entry(p, c) * v;
    }
}

/// `v ← (I − P) v`, `P` projecting onto `basis`, given `fit = Bᵀv`.
#[inline(always)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
pub(crate) fn project_twice(
    basis: &impl DiagonalGramBasis,
    v: &mut [f64],
    fit: &mut [f64],
    step: &mut [f64],
) {
    correct_twice(basis, v, fit, step);
    for (p, x) in v.iter_mut().enumerate() {
        *x = remove_fit(basis, p, *x, step);
    }
}

/// `fit ← G⁻¹Bᵀv` given `fit = Bᵀv`; `v` is left partly projected.
#[inline(always)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
pub(crate) fn fit_twice(
    basis: &impl DiagonalGramBasis,
    v: &mut [f64],
    fit: &mut [f64],
    step: &mut [f64],
) {
    correct_twice(basis, v, fit, step);
    fit.iter_mut().zip(step.iter()).for_each(|(f, s)| *f += s);
}

/// `fit ← G⁻¹Bᵀv`, `v` less that fit and `step` the fit of what remains.
///
/// A second classical Gram–Schmidt pass restores the digits the first rounds away when `v`
/// lies close to the span ("twice is enough"). For an intercept column this is the corrected
/// two-pass mean, weighted.
///
/// References:
/// - B. N. Parlett, *The Symmetric Eigenvalue Problem*, Prentice-Hall, 1980; SIAM reprint
///   1998, doi:10.1137/1.9781611971163 (Kahan's "twice is enough").
/// - J. W. Daniel, W. B. Gragg, L. Kaufman, G. W. Stewart, "Reorthogonalization and stable
///   algorithms for updating the Gram–Schmidt QR factorization", Math. Comp. 30(136),
///   772–795, 1976, doi:10.1090/S0025-5718-1976-0431641-8.
/// - L. Giraud, J. Langou, M. Rozložník, J. van den Eshof, "Rounding error analysis of the
///   classical Gram–Schmidt orthogonalization process", Numer. Math. 101, 87–100, 2005,
///   doi:10.1007/s00211-005-0615-4.
/// - T. F. Chan, G. H. Golub, R. J. LeVeque, "Algorithms for computing the sample variance:
///   analysis and recommendations", Amer. Statist. 37(3), 242–247, 1983,
///   doi:10.1080/00031305.1983.10483115.
#[inline(always)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
fn correct_twice(basis: &impl DiagonalGramBasis, v: &mut [f64], fit: &mut [f64], step: &mut [f64]) {
    solve_rescued(basis, v, fit);
    step.fill(0.0);
    for (p, x) in v.iter_mut().enumerate() {
        *x = remove_fit(basis, p, *x, fit);
        add_moments(basis, p, *x, step);
    }
    solve_rescued(basis, v, step);
}

/// `moments ← G⁻¹ moments`, `v`'s moments refolded at a power-of-two scale if any overflowed.
#[inline(always)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
fn solve_rescued(basis: &impl DiagonalGramBasis, v: &[f64], moments: &mut [f64]) {
    let overflowed = moments.iter().any(|m| !m.is_finite());
    let scale = if overflowed {
        rescale_moments(basis, v, moments)
    } else {
        1.0
    };
    for (c, m) in moments.iter_mut().enumerate() {
        let g = basis.norm2(c);
        *m = if g > 0.0 { *m / g } else { 0.0 };
    }
    if overflowed {
        moments.iter_mut().for_each(|m| *m /= scale);
    }
}

/// Refolds `v`'s moments at the power of two that keeps them finite, and returns it.
#[cold]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
fn rescale_moments(basis: &impl DiagonalGramBasis, v: &[f64], moments: &mut [f64]) -> f64 {
    let max_abs = v.iter().fold(0.0, |m: f64, x| m.max(x.abs()));
    let max_norm2 = (0..moments.len()).fold(0.0, |m: f64, c| m.max(basis.norm2(c)));
    // Cauchy–Schwarz: |Σ_p a_pc v_p| ≤ √(g_c · len) · max|v|.
    let log2 = max_abs.log2() + 0.5 * ((v.len() as f64).log2() + max_norm2.log2());
    let scale = 2f64.powi(1020 - log2.ceil() as i32);
    moments.fill(0.0);
    for (p, &x) in v.iter().enumerate() {
        add_moments(basis, p, scale * x, moments);
    }
    scale
}

/// `v − Σ_c entry(p, c) · fit_c`.
#[inline(always)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
fn remove_fit(basis: &impl DiagonalGramBasis, p: usize, v: f64, fit: &[f64]) -> f64 {
    (fit.iter().enumerate()).fold(v, |v, (c, m)| sub_product(v, basis.entry(p, c), *m))
}

/// `v − a · b`, fused where the target has the instruction; without it `mul_add` calls libm.
#[inline(always)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`System::Reduced` wires the reduced operator in")
)]
fn sub_product(v: f64, a: f64, b: f64) -> f64 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        (-a).mul_add(b, v)
    } else {
        v - a * b
    }
}

pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&x, &y)| x * y).sum()
}

/// Relative rank tolerance: the residual share of its variance below which a column drops.
const RANK_TOL: f64 = 1e-10;

/// Orthonormalizes `v×v` Gramians; reused, a sweep over Gramians allocates nothing.
pub(crate) struct GramBasisWorkspace {
    gram: Vec<f64>,
    residual: Vec<f64>,
    q: Vec<f64>,
    rows: Vec<f64>,
    kept: Vec<bool>,
}

/// Rows `W` in pivot order (row-major `rank × v`, `W·G·Wᵀ = I`); a dropped column is zero in `W`.
pub(crate) struct GramBasis<'a> {
    pub(crate) rows: &'a [f64],
    pub(crate) kept: &'a [bool],
}

impl GramBasisWorkspace {
    pub(crate) fn new(v: usize) -> Self {
        Self {
            gram: vec![0.0; v * v],
            residual: vec![0.0; v],
            q: vec![0.0; v],
            rows: Vec::with_capacity(v * v),
            kept: vec![false; v],
        }
    }

    /// Orthonormalize the row-major Gramian `fill` writes, by pivoted Gram–Schmidt.
    pub(crate) fn orthonormalize(&mut self, fill: impl FnOnce(&mut [f64])) -> GramBasis<'_> {
        let Self {
            gram,
            residual,
            q,
            rows,
            kept,
        } = self;
        let v = kept.len();
        fill(gram);
        for (r, j) in residual.iter_mut().zip(0..v) {
            *r = gram[j * v + j];
        }
        kept.fill(false);
        rows.clear();
        while let Some(p) = (0..v)
            .filter(|&j| {
                !kept[j] && residual[j].is_finite() && residual[j] > RANK_TOL * gram[j * v + j]
            })
            .max_by(|&a, &b| residual[a].total_cmp(&residual[b]))
        {
            kept[p] = true;

            // q = e_p − Σₜ ⟨e_p, qₜ⟩·qₜ, whose norm is already √residual[p].
            q.fill(0.0);
            q[p] = 1.0;
            for q_t in rows.chunks_exact(v) {
                let c = dot(&gram[p * v..][..v], q_t);
                for (qj, &qtj) in q.iter_mut().zip(q_t) {
                    *qj -= c * qtj;
                }
            }
            let norm = residual[p].sqrt();
            for qj in q.iter_mut() {
                *qj /= norm;
            }

            // Pythagoras: projecting out q costs every column ⟨e_j, q⟩² of variance.
            for (r, g_row) in residual.iter_mut().zip(gram.chunks_exact(v)) {
                let c = dot(g_row, q);
                *r -= c * c;
            }
            rows.extend_from_slice(q);
        }
        GramBasis { rows, kept }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthonormalizes_under_a_non_monotonic_pivot_order() {
        // Diagonals [2, 5, 3] force the pivot sequence 1 → 2 → 0, breaking order assumptions.
        let g = [2.0, 1.0, 0.5, 1.0, 5.0, 2.0, 0.5, 2.0, 3.0];
        let mut workspace = GramBasisWorkspace::new(3);
        let GramBasis { rows: w, kept } = workspace.orthonormalize(|gram| gram.copy_from_slice(&g));
        assert_eq!(kept, [true; 3]);
        assert_eq!(w.len(), 9);

        for r in 0..3 {
            for s in 0..3 {
                let wgw: f64 = (0..3)
                    .flat_map(|j| (0..3).map(move |k| (j, k)))
                    .map(|(j, k)| w[r * 3 + j] * g[j * 3 + k] * w[s * 3 + k])
                    .sum();
                let expected = if r == s { 1.0 } else { 0.0 };
                assert!(
                    (wgw - expected).abs() < 1e-12,
                    "(W·G·Wᵀ)[{r}][{s}] = {wgw}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn a_dropped_column_is_zero_in_every_row() {
        // Both [1, 0] and [½, ½] satisfy W·G·Wᵀ = 1 here; only zero on the dropped column is valid.
        let g = [1.0, 1.0, 1.0, 1.0];
        let mut workspace = GramBasisWorkspace::new(2);
        let GramBasis { rows: w, kept } = workspace.orthonormalize(|gram| gram.copy_from_slice(&g));
        assert_eq!(kept.iter().filter(|&&k| k).count(), 1);
        for (j, _) in kept.iter().enumerate().filter(|(_, &k)| !k) {
            assert!(w.chunks_exact(2).all(|row| row[j] == 0.0));
        }
    }
}
