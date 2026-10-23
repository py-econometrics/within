use std::ops::Range;

use rayon::prelude::*;
use schwarz_precond::Operator;

use super::Elimination;
use crate::SolveError;

/// Rows per parallel task, cut at an eliminated level's end so no level spans two tasks.
const ROWS_PER_TASK: usize = 16_384;

/// `(I − P_e) W^{1/2} D_kept` with rows grouped by eliminated level, so each product is one fused sweep.
pub(super) struct SegmentedDesign {
    /// Grouped row `p` is original row `order[p]`.
    order: Vec<u32>,
    /// Eliminated level `e` owns grouped rows `starts[e]..starts[e + 1]`.
    starts: Vec<usize>,
    /// The eliminated term's coefficient columns.
    width: usize,
    eliminated_intercept: bool,
    inverse: Vec<f64>,
    /// Each grouped row's eliminated slope loadings, row-major; an intercept's `1` is implicit.
    eliminated: Vec<f64>,
    /// Each grouped row's reduced coefficient per kept channel, intercepts first, row-major.
    columns: Vec<u32>,
    /// Each grouped row's loading per kept slope channel, row-major in `columns`' order.
    values: Vec<f64>,
    n_channels: usize,
    n_intercepts: usize,
    n_kept: usize,
    sqrt_weights: Option<Vec<f64>>,
    /// Eliminated levels per task.
    tasks: Vec<Range<usize>>,
}

/// One level's eliminated-column moments `Σ w a v` and their fit `G⁻ Σ w a v`.
struct LevelFit {
    sums: Vec<f64>,
    fit: Vec<f64>,
}

impl SegmentedDesign {
    pub(super) fn build(e: &Elimination<'_>) -> Self {
        let levels = e.levels();
        let n_levels = e.n_levels();
        let mut starts = vec![0usize; n_levels + 1];
        for &l in levels {
            starts[l as usize + 1] += 1;
        }
        for l in 0..n_levels {
            starts[l + 1] += starts[l];
        }
        let mut next = starts[..n_levels].to_vec();
        let mut order = vec![0u32; levels.len()];
        for (i, &l) in levels.iter().enumerate() {
            order[next[l as usize]] = i as u32;
            next[l as usize] += 1;
        }
        let (intercepts, slopes): (Vec<_>, Vec<_>) =
            (e.kept_channels().into_iter()).partition(|channel| channel.loading.is_none());
        let kept: Vec<_> = intercepts.iter().chain(&slopes).collect();
        let columns = order
            .par_iter()
            .flat_map_iter(|&i| {
                (kept.iter()).map(move |channel| channel.start as u32 + channel.levels[i as usize])
            })
            .collect();
        let slope_loadings: Vec<&[f64]> = slopes.iter().filter_map(|c| c.loading).collect();
        let values = order
            .par_iter()
            .flat_map_iter(|&i| slope_loadings.iter().map(move |z| z[i as usize]))
            .collect();
        let eliminated_loadings = e.eliminated_loadings();
        let eliminated_slopes: Vec<&[f64]> =
            eliminated_loadings.iter().flatten().copied().collect();
        let eliminated = order
            .par_iter()
            .flat_map_iter(|&i| eliminated_slopes.iter().map(move |z| z[i as usize]))
            .collect();
        let sqrt_weights = (e.prepared.sqrt_weights())
            .map(|sw| order.par_iter().map(|&i| sw[i as usize]).collect());
        let mut tasks = Vec::new();
        let mut first = 0;
        for l in 0..n_levels {
            if starts[l + 1] - starts[first] >= ROWS_PER_TASK {
                tasks.push(first..l + 1);
                first = l + 1;
            }
        }
        if first < n_levels {
            tasks.push(first..n_levels);
        }
        Self {
            order,
            starts,
            width: e.width,
            eliminated_intercept: eliminated_loadings[0].is_none(),
            inverse: e.inverse.clone(),
            eliminated,
            columns,
            values,
            n_channels: kept.len(),
            n_intercepts: intercepts.len(),
            n_kept: e.n_kept(),
            sqrt_weights,
            tasks,
        }
    }

    /// `b` in grouped row order, projected off the eliminated factor.
    pub(super) fn projected_rhs(&self, b: &[f64]) -> Vec<f64> {
        let mut grouped: Vec<f64> = self.order.par_iter().map(|&i| b[i as usize]).collect();
        let pieces = self.split(&mut grouped);
        pieces
            .into_par_iter()
            .for_each(|(levels, y)| match &self.sqrt_weights {
                Some(sw) => self.center(levels, y, |p| sw[p]),
                None => self.center(levels, y, |_| 1.0),
            });
        grouped
    }

    /// Each task's levels with its rows of `y`.
    fn split<'y>(&self, mut y: &'y mut [f64]) -> Vec<(Range<usize>, &'y mut [f64])> {
        let mut pieces = Vec::with_capacity(self.tasks.len());
        for levels in &self.tasks {
            let len = self.starts[levels.end] - self.starts[levels.start];
            let (head, tail) = std::mem::take(&mut y).split_at_mut(len);
            pieces.push((levels.clone(), head));
            y = tail;
        }
        pieces
    }

    fn level_fit(&self) -> LevelFit {
        LevelFit {
            sums: vec![0.0; self.width],
            fit: vec![0.0; self.width],
        }
    }

    fn eliminated_slopes(&self, p: usize) -> &[f64] {
        let n = self.width - self.eliminated_intercept as usize;
        &self.eliminated[p * n..][..n]
    }

    /// `f.sums += a_p v`.
    #[inline(always)]
    fn accumulate(&self, p: usize, v: f64, f: &mut LevelFit) {
        let slopes = match self.eliminated_intercept {
            true => {
                f.sums[0] += v;
                &mut f.sums[1..]
            }
            false => &mut f.sums[..],
        };
        for (sum, &a) in slopes.iter_mut().zip(self.eliminated_slopes(p)) {
            *sum += a * v;
        }
    }

    /// `f.fit ← G_l⁻ f.sums`, then `f.sums` cleared for the next level.
    #[inline(always)]
    fn solve(&self, l: usize, f: &mut LevelFit) {
        let c = self.width;
        let inverse = &self.inverse[l * c * c..][..c * c];
        for (fit, row) in f.fit.iter_mut().zip(inverse.chunks_exact(c)) {
            *fit = row.iter().zip(&f.sums).map(|(g, s)| g * s).sum();
        }
        f.sums.fill(0.0);
    }

    /// `a_pᵀ f.fit`.
    #[inline(always)]
    fn fitted(&self, p: usize, f: &LevelFit) -> f64 {
        let (head, slopes) = match self.eliminated_intercept {
            true => (f.fit[0], &f.fit[1..]),
            false => (0.0, &f.fit[..]),
        };
        (slopes.iter().zip(self.eliminated_slopes(p))).fold(head, |acc, (b, a)| acc + a * b)
    }

    /// `D_kept x` at grouped row `p`.
    #[inline(always)]
    fn row(&self, p: usize, x: &[f64]) -> f64 {
        let columns = &self.columns[p * self.n_channels..][..self.n_channels];
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        let unit: f64 = intercepts.iter().map(|&c| x[c as usize]).sum();
        (slopes.iter().zip(values)).fold(unit, |acc, (&c, &z)| acc + z * x[c as usize])
    }

    /// `acc += v D_kept[p, :]`.
    #[inline(always)]
    fn add_row(&self, p: usize, v: f64, acc: &mut [f64]) {
        let columns = &self.columns[p * self.n_channels..][..self.n_channels];
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        intercepts.iter().for_each(|&c| acc[c as usize] += v);
        (slopes.iter().zip(values)).for_each(|(&c, &z)| acc[c as usize] += z * v);
    }

    /// `y ← (I − P_e) y` on `levels`' rows, `y` starting at the first of them.
    #[inline(always)]
    fn center(&self, levels: Range<usize>, y: &mut [f64], w: impl Fn(usize) -> f64) {
        let base = self.starts[levels.start];
        let mut f = self.level_fit();
        for l in levels {
            let rows = self.starts[l]..self.starts[l + 1];
            (rows.clone()).for_each(|p| self.accumulate(p, w(p) * y[p - base], &mut f));
            self.solve(l, &mut f);
            rows.for_each(|p| y[p - base] -= w(p) * self.fitted(p, &f));
        }
    }

    #[inline(always)]
    fn gather(&self, x: &[f64], levels: Range<usize>, y: &mut [f64], w: impl Fn(usize) -> f64) {
        let base = self.starts[levels.start];
        let mut f = self.level_fit();
        for l in levels {
            let rows = self.starts[l]..self.starts[l + 1];
            for p in rows.clone() {
                let v = w(p) * self.row(p, x);
                y[p - base] = v;
                self.accumulate(p, w(p) * v, &mut f);
            }
            self.solve(l, &mut f);
            rows.for_each(|p| y[p - base] -= w(p) * self.fitted(p, &f));
        }
    }

    #[inline(always)]
    fn scatter(&self, u: &[f64], levels: Range<usize>, acc: &mut [f64], w: impl Fn(usize) -> f64) {
        let mut f = self.level_fit();
        for l in levels {
            let rows = self.starts[l]..self.starts[l + 1];
            (rows.clone()).for_each(|p| self.accumulate(p, w(p) * u[p], &mut f));
            self.solve(l, &mut f);
            for p in rows {
                let v = w(p) * (u[p] - w(p) * self.fitted(p, &f));
                self.add_row(p, v, acc);
            }
        }
    }
}

impl Operator for SegmentedDesign {
    fn nrows(&self) -> usize {
        self.order.len()
    }
    fn ncols(&self) -> usize {
        self.n_kept
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let pieces = self.split(y);
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
        let sum = (self.tasks.par_iter())
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

    fn assert_reproduces_reference(solver: &Solver<'_>) {
        let n = solver.prepared.design.n_obs;
        let e = Elimination::new(&solver.prepared);
        let (reference, fast) = (ProjectedDesign::new(&e), SegmentedDesign::build(&e));
        assert!(fast.tasks.len() > 1);
        let x: Vec<f64> = (0..fast.ncols()).map(|i| (i as f64).sin()).collect();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let (mut expected, mut got) = (vec![0.0; n], vec![0.0; n]);
        reference.apply(&x, &mut expected).unwrap();
        fast.apply(&x, &mut got).unwrap();
        for (p, &i) in fast.order.iter().enumerate() {
            assert!((got[p] - expected[i as usize]).abs() < 1e-11);
        }
        let grouped: Vec<f64> = fast.order.iter().map(|&i| u[i as usize]).collect();
        let (mut expected, mut got) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        reference.apply_adjoint(&u, &mut expected).unwrap();
        fast.apply_adjoint(&grouped, &mut got).unwrap();
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()));
        }
        let mut projected = u.clone();
        e.project(&mut projected, &mut vec![0.0; e.removed.len()]);
        let centered = fast.projected_rhs(&u);
        for (p, &i) in fast.order.iter().enumerate() {
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
            assert_reproduces_reference(&solver);
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
                assert_reproduces_reference(&solver);
            }
        }
    }
}
