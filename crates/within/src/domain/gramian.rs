//! `G = AᵀA`, read from the solve-basis design and `W^{1/2}`: what preconditioners approximate.

use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use rayon::prelude::*;

use super::{CoefficientLayout, Design, PreparedTerm, RowWeights};
use crate::channel::ChannelPair;
use crate::linalg::add_compensated;

/// The normal matrix `G = AᵀA` for `A = W^{1/2} D̃`.
#[derive(Clone, Copy)]
pub(crate) struct Gramian<'p> {
    /// `D̃`: the design in the solve basis.
    pub(super) design: &'p Design<'p>,
    pub(super) weights: &'p RowWeights,
}

impl<'p> Gramian<'p> {
    /// The coefficient space `G` acts on.
    pub(crate) fn layout(&self) -> &'p CoefficientLayout {
        &self.design.layout
    }

    /// `diag(G)` over term `term`'s coefficients, laid out like its `dofs()`, summed in observation order.
    pub(crate) fn term_diagonal(&self, term: usize) -> Vec<f64> {
        let prepared = self.design.term(term);
        let (layout, weights) = (prepared.layout, self.weights);
        let levels = prepared.rows.levels();
        let mut diag = vec![0.0; layout.n_dofs()];
        let mut corrections = vec![0.0; layout.n_dofs()];
        if !prepared.slopes.is_empty() {
            let columns: Vec<(usize, Option<&[f64]>)> = (0..layout.n_columns())
                .map(|c| (layout.block_column(c).start, prepared.loading(c)))
                .collect();
            for (obs, &level) in levels.iter().enumerate() {
                let w = weights.weight(obs);
                for &(base, z) in &columns {
                    // Keep `w * z * z` left-to-right: a zero weight kills a huge `z` first.
                    let slot = base + level as usize;
                    add_compensated(
                        &mut diag[slot],
                        &mut corrections[slot],
                        z.map_or(w, |z| w * z[obs] * z[obs]),
                    );
                }
            }
        } else if let Some(offsets) = prepared.rows.split.sorted_offsets() {
            // A sorted level is one run: a register sum keeps its order with no store per row.
            for (level, run) in offsets.windows(2).enumerate() {
                diag[level] += weights.sum(run[0]..run[1]);
            }
        } else {
            for (obs, &level) in levels.iter().enumerate() {
                let level = level as usize;
                add_compensated(
                    &mut diag[level],
                    &mut corrections[level],
                    weights.weight(obs),
                );
            }
        }
        for (entry, correction) in diag.iter_mut().zip(corrections) {
            *entry += correction;
        }
        diag
    }

    /// `diag(G)` in solve coordinates.
    pub(crate) fn diagonal(&self) -> Vec<f64> {
        let blocks: Vec<Vec<f64>> = (0..self.design.n_factors())
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
        fn side<'p>(term: &PreparedTerm<'p>, column: usize) -> FactorWidth1<'p, f64> {
            let (n, levels) = (term.layout.n_levels(), term.rows.levels());
            match term.loading(column) {
                Some(z) => FactorWidth1::valued(n, levels, z),
                None => FactorWidth1::new(n, levels),
            }
            .expect("levels index their term")
        }
        let (left, right) = (side(&rows, pair.rows.column), side(&cols, pair.cols.column));
        match self.weights.sqrt() {
            Some(s) => BipartiteFactorWidth2::scaled(left, right, s),
            None => BipartiteFactorWidth2::new(left, right),
        }
        .expect("both sides read the same observations")
    }
}

#[cfg(test)]
mod tests {
    use crate::domain::{Design, PreparedDesign};

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
    fn observation_sums(prepared: &PreparedDesign<'_>) -> Vec<f64> {
        let mut entries = vec![Vec::new(); prepared.design.n_dofs()];
        for term in prepared.design.terms() {
            for column in 0..term.layout.n_columns() {
                let block = &mut entries[term.layout.column_dofs(column)];
                let z = term.loading(column);
                for (obs, &level) in term.rows.levels().iter().enumerate() {
                    let w = prepared.weights.weight(obs);
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
        // Nested in `f`, so it comes out sorted and is summed by runs.
        let p = [0u32, 1, 0, 1, 0, 1, 1, 1];
        let weights = [2.0, 0.0, 3.0, 0.1, 1.0, 7.0, 0.3, 5.0];
        for weights in [None, Some(&weights[..])] {
            let effects = vec![
                crate::Effect::new(&f, true, [&z0[..], &z1[..]]).unwrap(),
                crate::Effect::new(&g, true, []).unwrap(),
                crate::Effect::new(&h, false, [&zh[..]]).unwrap(),
                crate::Effect::new(&p, true, []).unwrap(),
            ];
            let prepared = PreparedDesign::new(Design::new(effects).unwrap(), weights).unwrap();
            let terms = &prepared.design.terms;
            assert!(
                terms[3].split.sorted_offsets().is_some()
                    && terms[1].split.sorted_offsets().is_none()
            );
            assert!(prepared.unidentified().next().is_some());
            assert_eq!(
                prepared.gramian().diagonal(),
                observation_sums(&prepared),
                "weights {weights:?}"
            );
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
            let prepared = PreparedDesign::new(Design::new(effects).unwrap(), weights).unwrap();
            let diag = prepared.gramian().diagonal();
            let block =
                |term: usize, column| &diag[prepared.design.layout.term(term).column_dofs(column)];
            assert_eq!(block(0, 0), f_sums);
            assert_eq!(block(1, 0), g_sums);
            // A kept whitened slope has unit weighted norm; a dropped one is an exact-zero column.
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
