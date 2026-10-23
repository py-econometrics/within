use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::Operator;

use super::{Elimination, LevelGroups};
use crate::SolveError;

/// `P_e` one eliminated level at a time: moments `Σ_p a_p v_p` in, the fit `a_pᵀ G_l⁻ Σ a v` out.
pub(super) trait LevelProjection: Send + Sync {
    type Fit;
    fn fit(&self) -> Self::Fit;
    fn accumulate(&self, p: usize, v: f64, fit: &mut Self::Fit);
    /// Turns the level's moments into its coefficients, clearing the moments for the next level.
    fn solve(&self, l: usize, fit: &mut Self::Fit);
    fn fitted(&self, p: usize, fit: &Self::Fit) -> f64;
}

/// An intercept-only eliminated term: `P_e` subtracts weighted level means.
pub(super) struct LevelMeans {
    inverse: Vec<f64>,
}

/// An eliminated term with slopes: `P_e` subtracts each level's weighted least-squares fit.
pub(super) struct LevelFits {
    width: usize,
    intercept: bool,
    /// Per level, the generalized inverse of its `width × width` Gram block, row-major.
    inverse: Vec<f64>,
    /// Each grouped row's slope loadings, row-major; an intercept's `1` is implicit.
    slopes: Vec<f64>,
}

/// One level's moments and coefficients.
pub(super) struct BlockFit {
    sums: Vec<f64>,
    coefficients: Vec<f64>,
}

impl LevelProjection for LevelMeans {
    type Fit = (f64, f64);
    fn fit(&self) -> (f64, f64) {
        (0.0, 0.0)
    }
    #[inline(always)]
    fn accumulate(&self, _: usize, v: f64, (sum, _): &mut (f64, f64)) {
        *sum += v;
    }
    #[inline(always)]
    fn solve(&self, l: usize, (sum, mean): &mut (f64, f64)) {
        *mean = *sum * self.inverse[l];
        *sum = 0.0;
    }
    #[inline(always)]
    fn fitted(&self, _: usize, &(_, mean): &(f64, f64)) -> f64 {
        mean
    }
}

impl LevelFits {
    fn row(&self, p: usize) -> &[f64] {
        let n = self.width - self.intercept as usize;
        &self.slopes[p * n..][..n]
    }
}

impl LevelProjection for LevelFits {
    type Fit = BlockFit;
    fn fit(&self) -> BlockFit {
        BlockFit {
            sums: vec![0.0; self.width],
            coefficients: vec![0.0; self.width],
        }
    }
    #[inline(always)]
    fn accumulate(&self, p: usize, v: f64, f: &mut BlockFit) {
        let slopes = match self.intercept {
            true => {
                f.sums[0] += v;
                &mut f.sums[1..]
            }
            false => &mut f.sums[..],
        };
        for (sum, &a) in slopes.iter_mut().zip(self.row(p)) {
            *sum += a * v;
        }
    }
    #[inline(always)]
    fn solve(&self, l: usize, f: &mut BlockFit) {
        let c = self.width;
        let inverse = &self.inverse[l * c * c..][..c * c];
        for (b, row) in f.coefficients.iter_mut().zip(inverse.chunks_exact(c)) {
            *b = row.iter().zip(&f.sums).map(|(g, s)| g * s).sum();
        }
        f.sums.fill(0.0);
    }
    #[inline(always)]
    fn fitted(&self, p: usize, f: &BlockFit) -> f64 {
        let (head, slopes) = match self.intercept {
            true => (f.coefficients[0], &f.coefficients[1..]),
            false => (0.0, &f.coefficients[..]),
        };
        (slopes.iter().zip(self.row(p))).fold(head, |acc, (b, a)| acc + a * b)
    }
}

/// The kept design's rows in grouped order: unit intercept columns, then valued slope columns.
struct KeptRows {
    /// Each row's reduced coefficient per kept channel, intercepts first, row-major.
    columns: Vec<u32>,
    /// Each row's loading per kept slope channel, row-major in `columns`' order.
    values: Vec<f64>,
    n_channels: usize,
    n_intercepts: usize,
}

impl KeptRows {
    /// `D_kept x` at grouped row `p`.
    #[inline(always)]
    fn row(&self, p: usize, x: &[f64]) -> f64 {
        let columns = &self.columns[p * self.n_channels..][..self.n_channels];
        if self.values.is_empty() {
            return columns.iter().map(|&c| x[c as usize]).sum();
        }
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        let unit: f64 = intercepts.iter().map(|&c| x[c as usize]).sum();
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        (slopes.iter().zip(values)).fold(unit, |acc, (&c, &z)| acc + z * x[c as usize])
    }

    /// Each `(column, loading)` of row `p`.
    #[inline(always)]
    fn for_each_entry(&self, p: usize, mut f: impl FnMut(u32, f64)) {
        let columns = &self.columns[p * self.n_channels..][..self.n_channels];
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        intercepts.iter().for_each(|&c| f(c, 1.0));
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        (slopes.iter().zip(values)).for_each(|(&c, &z)| f(c, z));
    }

    /// `acc += v D_kept[p, :]`.
    #[inline(always)]
    fn add_row(&self, p: usize, v: f64, acc: &mut [f64]) {
        let columns = &self.columns[p * self.n_channels..][..self.n_channels];
        if self.values.is_empty() {
            return columns.iter().for_each(|&c| acc[c as usize] += v);
        }
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        intercepts.iter().for_each(|&c| acc[c as usize] += v);
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        (slopes.iter().zip(values)).for_each(|(&c, &z)| acc[c as usize] += z * v);
    }
}

/// `(I − P_e) W^{1/2} D_kept` with rows grouped by eliminated level, so each product is one fused sweep.
pub(super) struct SegmentedDesign<P> {
    groups: Arc<LevelGroups>,
    projection: P,
    kept: KeptRows,
    n_kept: usize,
    sqrt_weights: Option<Vec<f64>>,
}

/// The reduced design, its per-level projection fixed by the eliminated term's columns.
pub(super) enum ReducedDesign {
    Means(SegmentedDesign<LevelMeans>),
    Fits(SegmentedDesign<LevelFits>),
}

impl ReducedDesign {
    pub(super) fn build(e: &Elimination<'_>) -> Self {
        let groups = e.groups.clone();
        let loadings = e.eliminated_loadings();
        let slopes: Vec<&[f64]> = loadings.iter().flatten().copied().collect();
        let inverse = e.inverse.clone();
        match slopes.is_empty() {
            true => Self::Means(SegmentedDesign::build(e, groups, |_| LevelMeans {
                inverse,
            })),
            false => Self::Fits(SegmentedDesign::build(e, groups, |groups| LevelFits {
                width: e.width,
                intercept: loadings[0].is_none(),
                inverse,
                slopes: groups.gather(&slopes),
            })),
        }
    }

    /// `b` in grouped row order, projected off the eliminated factor.
    pub(super) fn projected_rhs(&self, b: &[f64]) -> Vec<f64> {
        match self {
            Self::Means(d) => d.projected_rhs(b),
            Self::Fits(d) => d.projected_rhs(b),
        }
    }

    pub(super) fn gram_diagonal(&self) -> Vec<f64> {
        match self {
            Self::Means(d) => d.gram_diagonal(),
            Self::Fits(d) => d.gram_diagonal(),
        }
    }

    #[cfg(test)]
    fn groups(&self) -> &LevelGroups {
        match self {
            Self::Means(d) => &d.groups,
            Self::Fits(d) => &d.groups,
        }
    }
}

impl<P: LevelProjection> SegmentedDesign<P> {
    fn build(
        e: &Elimination<'_>,
        groups: Arc<LevelGroups>,
        projection: impl FnOnce(&LevelGroups) -> P,
    ) -> Self {
        let (intercepts, slopes): (Vec<_>, Vec<_>) =
            (e.kept_channels().into_iter()).partition(|channel| channel.loading.is_none());
        let kept: Vec<_> = intercepts.iter().chain(&slopes).collect();
        let columns = (groups.order.par_iter())
            .flat_map_iter(|&i| {
                (kept.iter()).map(move |channel| channel.start as u32 + channel.levels[i as usize])
            })
            .collect();
        let loadings: Vec<&[f64]> = slopes.iter().filter_map(|c| c.loading).collect();
        let values = groups.gather(&loadings);
        let sqrt_weights = (e.prepared.sqrt_weights())
            .map(|sw| groups.order.par_iter().map(|&i| sw[i as usize]).collect());
        Self {
            projection: projection(&groups),
            groups,
            kept: KeptRows {
                columns,
                values,
                n_channels: kept.len(),
                n_intercepts: intercepts.len(),
            },
            n_kept: e.n_kept(),
            sqrt_weights,
        }
    }

    fn projected_rhs(&self, b: &[f64]) -> Vec<f64> {
        let mut grouped: Vec<f64> = (self.groups.order.par_iter())
            .map(|&i| b[i as usize])
            .collect();
        let pieces = self.groups.split(&mut grouped);
        pieces
            .into_par_iter()
            .for_each(|(levels, y)| match &self.sqrt_weights {
                Some(sw) => self.center(levels, y, |p| sw[p]),
                None => self.center(levels, y, |_| 1.0),
            });
        grouped
    }

    /// `y ← (I − P_e) y` on `levels`' rows, `y` starting at the first of them.
    #[inline(always)]
    fn center(&self, levels: Range<usize>, y: &mut [f64], w: impl Fn(usize) -> f64) {
        let (starts, projection) = (&self.groups.starts, &self.projection);
        let base = starts[levels.start];
        let mut fit = projection.fit();
        for l in levels {
            let rows = starts[l]..starts[l + 1];
            (rows.clone()).for_each(|p| projection.accumulate(p, w(p) * y[p - base], &mut fit));
            projection.solve(l, &mut fit);
            rows.for_each(|p| y[p - base] -= w(p) * projection.fitted(p, &fit));
        }
    }

    /// `diag(S)`: each kept column's squared norm less its projection onto the eliminated factor.
    fn gram_diagonal(&self) -> Vec<f64> {
        let zeros = || (vec![0.0; self.n_kept], vec![0.0; self.n_kept]);
        let (diagonal, norms) = (self.groups.tasks.par_iter())
            .fold(zeros, |mut acc, levels| {
                match &self.sqrt_weights {
                    Some(sw) => self.diagonal(levels.clone(), &mut acc, |p| sw[p]),
                    None => self.diagonal(levels.clone(), &mut acc, |_| 1.0),
                }
                acc
            })
            .reduce(zeros, |(mut d, mut n), (d2, n2)| {
                d.iter_mut().zip(&d2).for_each(|(a, b)| *a += b);
                n.iter_mut().zip(&n2).for_each(|(a, b)| *a += b);
                (d, n)
            });
        // A column inside the eliminated span keeps roundoff `~ε‖u‖²`, not zero.
        (diagonal.iter().zip(&norms))
            .map(|(&d, &n)| if d > 1e-12 * n { d } else { 0.0 })
            .collect()
    }

    fn diagonal(
        &self,
        levels: Range<usize>,
        (diagonal, norms): &mut (Vec<f64>, Vec<f64>),
        w: impl Fn(usize) -> f64,
    ) {
        let (starts, projection) = (&self.groups.starts, &self.projection);
        let mut fit = projection.fit();
        let mut entries: Vec<(u32, usize, f64)> = Vec::new();
        for l in levels {
            entries.clear();
            for p in starts[l]..starts[l + 1] {
                self.kept
                    .for_each_entry(p, |c, z| entries.push((c, p, w(p) * z)));
            }
            entries.sort_unstable_by_key(|&(c, ..)| c);
            for column in entries.chunk_by(|a, b| a.0 == b.0) {
                (column.iter()).for_each(|&(_, p, u)| projection.accumulate(p, w(p) * u, &mut fit));
                projection.solve(l, &mut fit);
                let (squared, projected) = (column.iter())
                    .fold((0.0, 0.0), |(s, q), &(_, p, u)| {
                        (s + u * u, q + u * w(p) * projection.fitted(p, &fit))
                    });
                let c = column[0].0 as usize;
                norms[c] += squared;
                diagonal[c] += squared - projected;
            }
        }
    }

    #[inline(always)]
    fn gather(&self, x: &[f64], levels: Range<usize>, y: &mut [f64], w: impl Fn(usize) -> f64) {
        let (starts, projection) = (&self.groups.starts, &self.projection);
        let base = starts[levels.start];
        let mut fit = projection.fit();
        for l in levels {
            let rows = starts[l]..starts[l + 1];
            for p in rows.clone() {
                let v = w(p) * self.kept.row(p, x);
                y[p - base] = v;
                projection.accumulate(p, w(p) * v, &mut fit);
            }
            projection.solve(l, &mut fit);
            rows.for_each(|p| y[p - base] -= w(p) * projection.fitted(p, &fit));
        }
    }

    #[inline(always)]
    fn scatter(&self, u: &[f64], levels: Range<usize>, acc: &mut [f64], w: impl Fn(usize) -> f64) {
        let (starts, projection) = (&self.groups.starts, &self.projection);
        let mut fit = projection.fit();
        for l in levels {
            let rows = starts[l]..starts[l + 1];
            (rows.clone()).for_each(|p| projection.accumulate(p, w(p) * u[p], &mut fit));
            projection.solve(l, &mut fit);
            for p in rows {
                let v = w(p) * (u[p] - w(p) * projection.fitted(p, &fit));
                self.kept.add_row(p, v, acc);
            }
        }
    }
}

impl<P: LevelProjection> Operator for SegmentedDesign<P> {
    fn nrows(&self) -> usize {
        self.groups.order.len()
    }
    fn ncols(&self) -> usize {
        self.n_kept
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let pieces = self.groups.split(y);
        pieces
            .into_par_iter()
            .for_each(|(levels, y)| match &self.sqrt_weights {
                Some(sw) => self.gather(x, levels, y, |p| sw[p]),
                None => self.gather(x, levels, y, |_| 1.0),
            });
        Ok(())
    }

    fn apply_adjoint(&self, u: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let zeros = || vec![0.0; self.n_kept];
        let sum = (self.groups.tasks.par_iter())
            .fold(zeros, |mut acc, levels| {
                match &self.sqrt_weights {
                    Some(sw) => self.scatter(u, levels.clone(), &mut acc, |p| sw[p]),
                    None => self.scatter(u, levels.clone(), &mut acc, |_| 1.0),
                }
                acc
            })
            .reduce(zeros, |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                a
            });
        y.copy_from_slice(&sum);
        Ok(())
    }
}

impl Operator for ReducedDesign {
    fn nrows(&self) -> usize {
        match self {
            Self::Means(d) => d.nrows(),
            Self::Fits(d) => d.nrows(),
        }
    }
    fn ncols(&self) -> usize {
        match self {
            Self::Means(d) => d.ncols(),
            Self::Fits(d) => d.ncols(),
        }
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        match self {
            Self::Means(d) => d.apply(x, y),
            Self::Fits(d) => d.apply(x, y),
        }
    }
    fn apply_adjoint(&self, u: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        match self {
            Self::Means(d) => d.apply_adjoint(u, y),
            Self::Fits(d) => d.apply_adjoint(u, y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ProjectedDesign;
    use super::*;
    use crate::{Design, Effect, PreconditionerConfig, Solver};
    use ndarray::Array2;

    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    0.2 + (i % 3) as f64
                }
            })
            .collect()
    }

    fn assert_reproduces_reference(solver: &Solver<'_>, slopes: bool) {
        let n = solver.prepared.design.n_obs;
        let e = Elimination::new(&solver.prepared);
        let (reference, fast) = (ProjectedDesign::new(&e), ReducedDesign::build(&e));
        assert_eq!(matches!(fast, ReducedDesign::Fits(_)), slopes);
        let order = &fast.groups().order;
        assert!(fast.groups().tasks.len() > 1);
        let x: Vec<f64> = (0..fast.ncols()).map(|i| (i as f64).sin()).collect();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let (mut expected, mut got) = (vec![0.0; n], vec![0.0; n]);
        reference.apply(&x, &mut expected).unwrap();
        fast.apply(&x, &mut got).unwrap();
        for (p, &i) in order.iter().enumerate() {
            assert!((got[p] - expected[i as usize]).abs() < 1e-11);
        }
        let grouped: Vec<f64> = order.iter().map(|&i| u[i as usize]).collect();
        let (mut expected, mut got) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        reference.apply_adjoint(&u, &mut expected).unwrap();
        fast.apply_adjoint(&grouped, &mut got).unwrap();
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()));
        }
        let diagonal = fast.gram_diagonal();
        let mut column = vec![0.0; n];
        for (j, &d) in diagonal.iter().enumerate() {
            let unit: Vec<f64> = (0..x.len()).map(|i| (i == j) as u8 as f64).collect();
            reference.apply(&unit, &mut column).unwrap();
            let squared: f64 = column.iter().map(|v| v * v).sum();
            assert!(
                (d - squared).abs() < 1e-9 * (1.0 + squared),
                "column {j}: {d} vs {squared}"
            );
        }
        let mut projected = u.clone();
        e.project(&mut projected, &mut vec![0.0; e.removed.len()]);
        let centered = fast.projected_rhs(&u);
        for (p, &i) in order.iter().enumerate() {
            assert!((centered[p] - projected[i as usize]).abs() < 1e-11);
        }
    }

    #[test]
    fn grouped_rows_reproduce_the_projected_design() {
        let n = 30_000;
        let categories = Array2::from_shape_fn((n, 3), |(i, j)| {
            ((i * (2 * j + 1) + i / 7) % [5003, 131, 7][j]) as u32
        });
        let weights = weights(n);
        for weights in [None, Some(&weights[..])] {
            let design = Design::from_categories(categories.view()).unwrap();
            let solver = Solver::new(design, weights, PreconditionerConfig::Diagonal).unwrap();
            assert_reproduces_reference(&solver, false);
        }
    }

    #[test]
    fn grouped_rows_reproduce_the_projected_slope_design() {
        let n = 30_000;
        let worker: Vec<u32> = (0..n).map(|i| (i % 3001) as u32).collect();
        let firm: Vec<u32> = (0..n).map(|i| ((i * 7 + i / 11) % 131) as u32).collect();
        let year: Vec<u32> = (0..n).map(|i| (i / 3001 % 10) as u32).collect();
        let z1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let z2: Vec<f64> = (0..n)
            .map(|i| year[i] as f64 + 1e-3 * (i as f64).cos())
            .collect();
        let z3: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos() + 2.0).collect();
        let weights = weights(n);
        for intercept in [true, false] {
            for weights in [None, Some(&weights[..])] {
                let effects = vec![
                    Effect::new(&worker, intercept, [&z1[..], &z2[..]]).unwrap(),
                    Effect::new(&firm, true, [&z3[..]]).unwrap(),
                    Effect::new(&year, true, []).unwrap(),
                ];
                let solver = Solver::new(effects, weights, PreconditionerConfig::Diagonal).unwrap();
                assert_reproduces_reference(&solver, true);
            }
            // Kept slopes alone: the eliminated term is intercept-only.
            let effects = vec![
                Effect::new(&worker, true, []).unwrap(),
                Effect::new(&firm, intercept, [&z3[..]]).unwrap(),
                Effect::new(&year, true, [&z1[..]]).unwrap(),
            ];
            let solver = Solver::new(effects, None, PreconditionerConfig::Diagonal).unwrap();
            assert_reproduces_reference(&solver, false);
        }
    }
}
