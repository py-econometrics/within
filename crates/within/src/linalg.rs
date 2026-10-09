//! Shared dense linear-algebra kernels.

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
