//! `G = AᵀA` for `A = W^{1/2} D̃`: its term blocks are diagonal, its cross blocks factored.

use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use rayon::prelude::*;

use super::{OrthogonalDesign, TermBlock};
use crate::design::ChannelPair;
use crate::design::CoefficientLayout;
use crate::linalg::add_compensated;

/// The normal matrix `G = AᵀA` for `A = W^{1/2} D̃`.
#[derive(Clone, Copy)]
pub(crate) struct Gramian<'p> {
    design: &'p OrthogonalDesign<'p>,
}

impl<'p> Gramian<'p> {
    /// `G = D̃ᵀ W D̃`.
    pub(crate) fn new(design: &'p OrthogonalDesign<'p>) -> Self {
        Self { design }
    }

    /// The coefficient space `G` acts on.
    pub(crate) fn layout(&self) -> &'p CoefficientLayout {
        self.design.layout()
    }

    /// `diag(G)` over term `term`'s coefficients, laid out like its `dofs()`. `D̃_tᵀ W D̃_t` is
    /// diagonal per level: `w_l` on the intercept, `1` on the identified slopes, `0` on the rest.
    fn term_diagonal(&self, term: usize) -> Vec<f64> {
        let layout = self.layout().term(term);
        let mut diag = vec![0.0; layout.n_dofs()];
        if let Some(column) = layout.intercept_column() {
            // `diag(E_tᵀ W E_t)`: per level `Σ w`, in the pieces the term's reduction fixes.
            let (rows, sums) = (
                self.design.incidence.term(term),
                &mut diag[layout.block_column(column)],
            );
            match self.design.weights.sqrt() {
                // A sum of ones is exact in any order.
                None => rows.fold(
                    0.0,
                    &|s: &mut f64, _, _| *s += 1.0,
                    &|l: &mut f64, r: &f64| *l += r,
                    sums,
                ),
                Some(_) => {
                    let weights = &self.design.weights;
                    let mut compensated = vec![(0.0, 0.0); sums.len()];
                    rows.fold(
                        (0.0, 0.0),
                        &|(s, c): &mut (f64, f64), _, obs| {
                            add_compensated(s, c, weights.weight(obs))
                        },
                        &|(s, c): &mut (f64, f64), &(rs, rc): &(f64, f64)| {
                            add_compensated(s, c, rs);
                            *c += rc;
                        },
                        &mut compensated[..],
                    );
                    for (d, (s, c)) in sums.iter_mut().zip(compensated) {
                        *d = s + c;
                    }
                }
            }
        }
        for (level, rank) in self.design.slopes[term].ranks().enumerate() {
            for j in 0..rank {
                diag[layout.block_column(layout.slope_column(j)).start + level] = 1.0;
            }
        }
        diag
    }

    /// `diag(G)` in solve coordinates.
    pub(crate) fn diagonal(&self) -> Vec<f64> {
        let blocks: Vec<Vec<f64>> = (0..self.layout().n_terms())
            .into_par_iter()
            .map(|term| self.term_diagonal(term))
            .collect();
        blocks.concat()
    }

    /// Pair `pair`'s block of `G` as its factor `V`: `G = V Vᵀ`, one column per observation.
    pub(crate) fn pair_factor(&self, pair: ChannelPair) -> BipartiteFactorWidth2<'p, f64> {
        let (rows, cols) = (
            self.design.term(pair.rows.term),
            self.design.term(pair.cols.term),
        );
        fn side<'p>(term: &TermBlock<'p>, column: usize) -> FactorWidth1<'p, f64> {
            let (n, levels) = (term.layout().n_levels(), term.levels());
            match term.loading(column) {
                Some(z) => FactorWidth1::valued(n, levels, z),
                None => FactorWidth1::new(n, levels),
            }
            .expect("levels index their term")
        }
        let (left, right) = (side(&rows, pair.rows.column), side(&cols, pair.cols.column));
        match self.design.weights.sqrt() {
            Some(s) => BipartiteFactorWidth2::scaled(left, right, s),
            None => BipartiteFactorWidth2::new(left, right),
        }
        .expect("both sides read the same observations")
    }
}

#[cfg(test)]
mod tests {
    use super::Gramian;
    use crate::design::{Channel, Design};
    use crate::system::OrthogonalDesign;

    /// Independent floating-point expansion: retain each exact two-sum remainder
    /// before rounding the short expansion, rather than reusing Neumaier's kernel.
    fn expansion_sum(values: Vec<f64>) -> f64 {
        let mut partials: Vec<f64> = Vec::new();
        for mut x in values {
            let mut kept = 0;
            for i in 0..partials.len() {
                let mut y = partials[i];
                if x.abs() < y.abs() {
                    std::mem::swap(&mut x, &mut y);
                }
                let hi = x + y;
                let lo = y - (hi - x);
                if lo != 0.0 {
                    partials[kept] = lo;
                    kept += 1;
                }
                x = hi;
            }
            partials.truncate(kept);
            partials.push(x);
        }
        let mut hi = partials.pop().unwrap_or(0.0);
        let mut lo = 0.0;
        while let Some(y) = partials.pop() {
            let x = hi;
            hi = x + y;
            lo = y - (hi - x);
            if lo != 0.0 {
                break;
            }
        }
        // A same-sign tail resolves a halfway rounding at the leading pair.
        if partials.last().is_some_and(|tail| lo * tail > 0.0) {
            let twice = 2.0 * lo;
            let rounded = hi + twice;
            if rounded - hi == twice {
                hi = rounded;
            }
        }
        hi
    }

    /// Explicit per-column, per-level contributions, independently summed.
    fn observation_sums(design: &OrthogonalDesign<'_>) -> Vec<f64> {
        let mut entries = vec![Vec::new(); design.n_dofs()];
        for (t, term) in design.terms().enumerate() {
            for column in 0..term.layout().n_columns() {
                let block = &mut entries[design.layout().dofs(Channel { term: t, column })];
                let z = term.loading(column);
                for (obs, &level) in term.levels().iter().enumerate() {
                    let w = design.weights.weight(obs);
                    block[level as usize].push(z.map_or(w, |z| w * z[obs] * z[obs]));
                }
            }
        }
        entries.into_iter().map(expansion_sum).collect()
    }

    #[test]
    fn gram_diagonal_matches_independent_expansion_sums() {
        // Level 1 of `f` has z1 = 2·z0 (a rank drop).
        let f = [0u32, 1, 0, 1, 0, 1, 2, 2];
        let z0 = [1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 0.3, 0.7];
        let z1 = [9.0, 4.0, 1.0, 8.0, 2.0, 12.0, 5.0, 1.1];
        let g = [2u32, 0, 1, 1, 0, 2, 0, 1];
        let h = [1u32, 0, 0, 1, 1, 0, 1, 0];
        let zh = [0.5, 1.5, -2.0, 3.0, 0.25, -1.0, 7.0, 2.0];
        // Nested in `f`, so it comes out sorted.
        let p = [0u32, 1, 0, 1, 0, 1, 1, 1];
        let weights = [2.0, 0.0, 3.0, 0.1, 1.0, 7.0, 0.3, 5.0];
        for weights in [None, Some(&weights[..])] {
            let effects = vec![
                crate::Effect::new(&f, true, [&z0[..], &z1[..]]).unwrap(),
                crate::Effect::new(&g, true, []).unwrap(),
                crate::Effect::new(&h, false, [&zh[..]]).unwrap(),
                crate::Effect::new(&p, true, []).unwrap(),
            ];
            let design = Design::new(effects)
                .unwrap()
                .orthogonalize(weights)
                .unwrap();
            assert!(design.unidentified().next().is_some());
            // Read off the level Grams, the diagonal agrees with the data's to rounding,
            // and is zero exactly where the data's is.
            let diag = Gramian::new(&design).diagonal();
            for (d, o) in diag.iter().zip(observation_sums(&design)) {
                assert_eq!(*d == 0.0, o == 0.0, "weights {weights:?}");
                assert!((d - o).abs() <= 1e-12 * o.abs().max(1.0), "{d} vs {o}");
            }
        }
    }

    #[test]
    fn gram_diagonal_matches_hand_computed_values() {
        // `z` is constant in level 1 of `f`, so that level's slope is dropped.
        let f = [0u32, 0, 0, 1, 1, 2, 2];
        let z = [1.0, 2.0, 4.0, 3.0, 3.0, 5.0, 7.0];
        let g = [0u32, 1, 0, 1, 0, 1, 1];
        // Perfect squares, so `(√w)²` is exact.
        let weights = [1.0, 4.0, 9.0, 0.25, 4.0, 1.0, 16.0];
        for (weights, f_sums, g_sums) in [
            (None, [3.0, 2.0, 2.0], [3.0, 4.0]),
            (Some(&weights[..]), [14.0, 4.25, 17.0], [14.0, 21.25]),
        ] {
            let effects = vec![
                crate::Effect::new(&f, true, [&z[..]]).unwrap(),
                crate::Effect::new(&g, true, []).unwrap(),
            ];
            let design = Design::new(effects)
                .unwrap()
                .orthogonalize(weights)
                .unwrap();
            let diag = Gramian::new(&design).diagonal();
            let block = |term: usize, column| &diag[design.layout().dofs(Channel { term, column })];
            assert_eq!(block(0, 0), f_sums);
            assert_eq!(block(1, 0), g_sums);
            // A kept orthogonalized slope has unit weighted norm; a dropped one is an exact-zero column.
            let slope = block(0, 1);
            assert_eq!(slope[1], 0.0, "weights {weights:?}");
            for level in [0, 2] {
                assert!(
                    (slope[level] - 1.0).abs() < 1e-12,
                    "{slope:?}, weights {weights:?}"
                );
            }
        }
    }
}
