use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::elimination::{Elimination, LevelGroups, RowOrder};
use crate::SolveError;

#[cfg(test)]
mod tests;

/// Rows per parallel task, cut at an eliminated level's end so no level spans two tasks.
const ROWS_PER_TASK: usize = 16_384;

/// Eliminated levels per parallel task.
struct LevelTasks(Vec<Range<usize>>);

impl LevelTasks {
    fn build(starts: &[usize]) -> Self {
        let n_levels = starts.len() - 1;
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
        Self(tasks)
    }

    /// Each task's levels with its rows of `y`, grouped rows of level `l` starting at `starts[l]`.
    fn for_each_mut(
        &self,
        starts: &[usize],
        y: &mut [f64],
        visit: impl Fn(Range<usize>, &mut [f64]) + Send + Sync,
    ) {
        if self.0.is_empty() {
            return;
        }
        rayon::iter::split((&self.0[..], y), |(tasks, y)| {
            if tasks.len() < 2 {
                return ((tasks, y), None);
            }
            let (left, right) = tasks.split_at(tasks.len() / 2);
            let split = starts[right[0].start] - starts[left[0].start];
            let (a, b) = y.split_at_mut(split);
            ((left, a), Some((right, b)))
        })
        .for_each(|(tasks, y)| visit(tasks[0].start..tasks[tasks.len() - 1].end, y));
    }

    fn batches(&self) -> impl IndexedParallelIterator<Item = Range<usize>> + '_ {
        // Bound coefficient-sized accumulators by the pool size, independent of row count.
        let chunk = self.0.len().div_ceil(rayon::current_num_threads()).max(1);
        self.0
            .par_chunks(chunk)
            .map(|tasks| tasks[0].start..tasks[tasks.len() - 1].end)
    }
}

/// `P_e` one eliminated level at a time: moments `Σ_p a_p v_p` in, the fit `a_pᵀ G_l⁻ Σ a v` out.
trait LevelProjection: Send + Sync {
    type Fit;
    fn fit(&self) -> Self::Fit;
    fn accumulate(&self, p: usize, v: f64, fit: &mut Self::Fit);
    /// Turns the level's moments into its coefficients, clearing the moments for the next level.
    fn solve(&self, l: usize, fit: &mut Self::Fit);
    fn fitted(&self, p: usize, fit: &Self::Fit) -> f64;
}

/// An intercept-only eliminated term: `P_e` subtracts weighted level means.
struct LevelMeans {
    inverse: Arc<Vec<f64>>,
}

/// An eliminated term with slopes: `P_e` subtracts each level's weighted least-squares fit.
struct LevelFits {
    width: usize,
    intercept: bool,
    /// Per level, the generalized inverse of its `width × width` Gram block, row-major.
    inverse: Arc<Vec<f64>>,
    /// Each grouped row's slope loadings, row-major; an intercept's `1` is implicit.
    slopes: Vec<f64>,
}

/// One level's moments and coefficients.
struct BlockFit {
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
    #[inline(always)]
    fn columns<const WIDTH: usize>(&self, p: usize) -> &[u32] {
        let width = if WIDTH == 0 { self.n_channels } else { WIDTH };
        &self.columns[p * width..][..width]
    }

    /// `D_kept x` at grouped row `p`.
    #[inline(always)]
    fn row<const WIDTH: usize>(&self, p: usize, x: &[f64]) -> f64 {
        let columns = self.columns::<WIDTH>(p);
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
        let columns = self.columns::<0>(p);
        let (intercepts, slopes) = columns.split_at(self.n_intercepts);
        intercepts.iter().for_each(|&c| f(c, 1.0));
        let values = &self.values[p * slopes.len()..][..slopes.len()];
        (slopes.iter().zip(values)).for_each(|(&c, &z)| f(c, z));
    }

    /// `acc += v D_kept[p, :]`.
    #[inline(always)]
    fn add_row(&self, p: usize, v: f64, acc: &mut [f64]) {
        let columns = self.columns::<0>(p);
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
struct SegmentedDesign<P> {
    groups: Arc<LevelGroups>,
    tasks: LevelTasks,
    projection: P,
    kept: KeptRows,
    n_kept: usize,
    sqrt_weights: Option<Vec<f64>>,
}

pub(crate) trait ReducedDesign: Operator {
    fn operator(&self) -> &dyn Operator;
    fn projected_rhs(&self, b: &[f64]) -> Vec<f64>;
    /// `diag(S)` and the kept columns' squared norms `diag(A_KK)`.
    fn gram_diagonal_and_norms(&self) -> (Vec<f64>, Vec<f64>);
    fn gram_diagonal(&self) -> Vec<f64> {
        self.gram_diagonal_and_norms().0
    }
    #[cfg(test)]
    fn groups(&self) -> &LevelGroups;
    #[cfg(test)]
    fn n_tasks(&self) -> usize;
}

pub(crate) fn build(e: &Elimination<'_>) -> Box<dyn ReducedDesign> {
    let loadings = e.eliminated_loadings();
    let slopes: Vec<&[f64]> = loadings.iter().flatten().copied().collect();
    let inverse = e.inverse().clone();
    if slopes.is_empty() {
        Box::new(SegmentedDesign::build(e, LevelMeans { inverse }))
    } else {
        Box::new(SegmentedDesign::build(
            e,
            LevelFits {
                width: e.width(),
                intercept: loadings[0].is_none(),
                inverse,
                slopes: e.groups().gather(&slopes),
            },
        ))
    }
}

impl<P: LevelProjection> SegmentedDesign<P> {
    fn build(e: &Elimination<'_>, projection: P) -> Self {
        let groups = e.groups().clone();
        let (intercepts, slopes): (Vec<_>, Vec<_>) =
            (e.kept_channels().into_iter()).partition(|channel| channel.loading.is_none());
        let kept: Vec<_> = intercepts.iter().chain(&slopes).collect();
        let mut columns = vec![0; groups.order.len() * kept.len()];
        if !kept.is_empty() {
            columns
                .par_chunks_mut(kept.len())
                .zip(groups.order.par_iter())
                .for_each(|(row, i)| {
                    row.iter_mut().zip(&kept).for_each(|(c, channel)| {
                        *c = channel.start as u32 + channel.levels[i as usize];
                    });
                });
        }
        let loadings: Vec<&[f64]> = slopes.iter().filter_map(|c| c.loading).collect();
        let values = groups.gather(&loadings);
        let sqrt_weights = (e.prepared().sqrt_weights())
            .map(|sw| groups.order.par_iter().map(|i| sw[i as usize]).collect());
        Self {
            projection,
            tasks: LevelTasks::build(&groups.starts),
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

    fn projected_rhs_rows(&self, row: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
        let mut grouped = vec![0.0; self.groups.order.len()];
        self.tasks
            .for_each_mut(&self.groups.starts, &mut grouped, |levels, y| {
                match &self.sqrt_weights {
                    Some(sw) => self.project_rows(levels, y, |p| sw[p], &row),
                    None => self.project_rows(levels, y, |_| 1.0, &row),
                }
            });
        grouped
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
    fn project_rows(
        &self,
        levels: Range<usize>,
        y: &mut [f64],
        w: impl Fn(usize) -> f64,
        row: impl Fn(usize) -> f64,
    ) {
        let (starts, projection) = (&self.groups.starts, &self.projection);
        let base = starts[levels.start];
        let mut fit = projection.fit();
        for l in levels {
            let rows = starts[l]..starts[l + 1];
            for p in rows.clone() {
                let v = row(p);
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
        match self.kept.n_channels {
            2 => self.apply_rows::<2>(x, y),
            _ => self.apply_rows::<0>(x, y),
        }
    }

    fn apply_adjoint(&self, u: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let scatter = |levels: Range<usize>, acc: &mut [f64]| match &self.sqrt_weights {
            Some(sw) => self.scatter(u, levels, acc, |p| sw[p]),
            None => self.scatter(u, levels, acc, |_| 1.0),
        };
        if self.tasks.0.len() <= 1 || rayon::current_num_threads() == 1 {
            y.fill(0.0);
            scatter(0..self.groups.starts.len() - 1, y);
            return Ok(());
        }
        let sum = self
            .tasks
            .batches()
            .map(|levels| {
                let mut acc = vec![0.0; self.n_kept];
                scatter(levels, &mut acc);
                acc
            })
            .reduce_with(|mut a, b| {
                a.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                a
            })
            .expect("nonempty parallel tasks");
        y.copy_from_slice(&sum);
        Ok(())
    }
}

impl<P: LevelProjection> SegmentedDesign<P> {
    fn apply_rows<const WIDTH: usize>(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.tasks
            .for_each_mut(&self.groups.starts, y, |levels, y| {
                match &self.sqrt_weights {
                    Some(sw) => self.project_rows(
                        levels,
                        y,
                        |p| sw[p],
                        |p| sw[p] * self.kept.row::<WIDTH>(p, x),
                    ),
                    None => self.project_rows(levels, y, |_| 1.0, |p| self.kept.row::<WIDTH>(p, x)),
                }
            });
        Ok(())
    }
}

impl<P: LevelProjection> ReducedDesign for SegmentedDesign<P> {
    fn operator(&self) -> &dyn Operator {
        self
    }

    fn projected_rhs(&self, b: &[f64]) -> Vec<f64> {
        match &self.groups.order {
            RowOrder::Identity(_) => self.projected_rhs_rows(|p| b[p]),
            RowOrder::Permutation(rows) => self.projected_rhs_rows(|p| b[rows[p] as usize]),
        }
    }

    /// `diag(S)`: each kept column's squared norm less its projection onto the eliminated factor.
    fn gram_diagonal_and_norms(&self) -> (Vec<f64>, Vec<f64>) {
        let zeros = || (vec![0.0; self.n_kept], vec![0.0; self.n_kept]);
        let (mut diagonal, norms) = self
            .tasks
            .batches()
            .map(|levels| {
                let mut acc = zeros();
                match &self.sqrt_weights {
                    Some(sw) => self.diagonal(levels, &mut acc, |p| sw[p]),
                    None => self.diagonal(levels, &mut acc, |_| 1.0),
                }
                acc
            })
            .reduce_with(|(mut d, mut n), (d2, n2)| {
                d.iter_mut().zip(&d2).for_each(|(a, b)| *a += b);
                n.iter_mut().zip(&n2).for_each(|(a, b)| *a += b);
                (d, n)
            })
            .unwrap_or_else(zeros);
        // A column inside the eliminated span keeps roundoff `~ε‖u‖²`, not zero.
        diagonal.iter_mut().zip(&norms).for_each(|(d, &n)| {
            *d = if *d > 1e-12 * n { *d } else { 0.0 };
        });
        (diagonal, norms)
    }

    #[cfg(test)]
    fn groups(&self) -> &LevelGroups {
        &self.groups
    }

    #[cfg(test)]
    fn n_tasks(&self) -> usize {
        self.tasks.0.len()
    }
}
