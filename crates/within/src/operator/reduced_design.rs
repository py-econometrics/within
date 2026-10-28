use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::elimination::{Elimination, GroupedRows, RowOrder};
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
    /// The level's solved coefficients, in eliminated-column order.
    fn coefficients<'f>(&self, fit: &'f Self::Fit) -> &'f [f64];
}

/// An intercept-only eliminated term: `P_e` subtracts weighted level means.
struct LevelMeans {
    inverse: Arc<Vec<f64>>,
}

/// An eliminated term with slopes: `P_e` subtracts each level's weighted least-squares fit.
struct LevelFits {
    /// Per level, the generalized inverse of its `width × width` Gram block, row-major.
    inverse: Arc<Vec<f64>>,
    rows: Arc<GroupedRows>,
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
    fn coefficients<'f>(&self, (_, mean): &'f (f64, f64)) -> &'f [f64] {
        std::slice::from_ref(mean)
    }
}

impl LevelProjection for LevelFits {
    type Fit = BlockFit;
    fn fit(&self) -> BlockFit {
        let width = self.rows.eliminated.width;
        BlockFit {
            sums: vec![0.0; width],
            coefficients: vec![0.0; width],
        }
    }
    #[inline(always)]
    fn accumulate(&self, p: usize, v: f64, f: &mut BlockFit) {
        let slopes = match self.rows.eliminated.intercept {
            true => {
                f.sums[0] += v;
                &mut f.sums[1..]
            }
            false => &mut f.sums[..],
        };
        for (sum, &a) in slopes.iter_mut().zip(self.rows.eliminated.slopes(p)) {
            *sum += a * v;
        }
    }
    #[inline(always)]
    fn solve(&self, l: usize, f: &mut BlockFit) {
        let c = self.rows.eliminated.width;
        let inverse = &self.inverse[l * c * c..][..c * c];
        for (b, row) in f.coefficients.iter_mut().zip(inverse.chunks_exact(c)) {
            *b = row.iter().zip(&f.sums).map(|(g, s)| g * s).sum();
        }
        f.sums.fill(0.0);
    }
    #[inline(always)]
    fn fitted(&self, p: usize, f: &BlockFit) -> f64 {
        let (head, slopes) = match self.rows.eliminated.intercept {
            true => (f.coefficients[0], &f.coefficients[1..]),
            false => (0.0, &f.coefficients[..]),
        };
        let row = self.rows.eliminated.slopes(p);
        (slopes.iter().zip(row)).fold(head, |acc, (b, a)| acc + a * b)
    }
    fn coefficients<'f>(&self, f: &'f BlockFit) -> &'f [f64] {
        &f.coefficients
    }
}

/// `(I − P_e) W^{1/2} D_kept` with rows grouped by eliminated level, so each product is one fused sweep.
struct SegmentedDesign<P> {
    rows: Arc<GroupedRows>,
    tasks: LevelTasks,
    projection: P,
    n_kept: usize,
}

/// Per kept coefficient: `diag(S)`, `diag(A_KK)`, and what eliminating one column alone removes.
pub(crate) struct ColumnMoments {
    pub(crate) diagonal: Vec<f64>,
    pub(crate) norms: Vec<f64>,
    /// Per eliminated column `j`: `Σ_e (Σ_{i∈e} w a x_j)² / Σ_{i∈e} w x_j²`.
    pub(crate) single_removal: Vec<Vec<f64>>,
}

pub(crate) trait ReducedDesign: Operator {
    fn operator(&self) -> &dyn Operator;
    fn projected_rhs(&self, b: &[f64]) -> Vec<f64>;
    fn column_moments(&self) -> ColumnMoments;
    fn gram_diagonal(&self) -> Vec<f64> {
        self.column_moments().diagonal
    }
    /// The eliminated coefficients fitting `b − W½ D_kept x`, laid out like the removed slots.
    fn recover(&self, b: &[f64], x: &[f64], coefficients: &mut [f64]);
    #[cfg(test)]
    fn n_tasks(&self) -> usize;
}

pub(crate) fn build(e: &Elimination<'_>) -> Box<dyn ReducedDesign> {
    let (rows, inverse) = (e.rows().clone(), e.inverse().clone());
    if rows.eliminated.width == 1 && rows.eliminated.intercept {
        Box::new(SegmentedDesign::build(e, LevelMeans { inverse }))
    } else {
        Box::new(SegmentedDesign::build(e, LevelFits { inverse, rows }))
    }
}

impl<P: LevelProjection> SegmentedDesign<P> {
    fn build(e: &Elimination<'_>, projection: P) -> Self {
        let rows = e.rows().clone();
        Self {
            projection,
            tasks: LevelTasks::build(&rows.starts),
            rows,
            n_kept: e.n_kept(),
        }
    }

    fn projected_rhs_rows(&self, row: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
        let mut grouped = vec![0.0; self.rows.order.len()];
        self.tasks
            .for_each_mut(&self.rows.starts, &mut grouped, |levels, y| {
                match &self.rows.sqrt_weights {
                    Some(sw) => self.project_rows(levels, y, |p| sw[p], &row),
                    None => self.project_rows(levels, y, |_| 1.0, &row),
                }
            });
        grouped
    }

    fn moments(&self, levels: Range<usize>, acc: &mut ColumnMoments, w: impl Fn(usize) -> f64) {
        let (starts, projection) = (&self.rows.starts, &self.projection);
        let eliminated = &self.rows.eliminated;
        let x = |p: usize, j: usize| w(p) * eliminated.loading(p, j);
        let mut fit = projection.fit();
        let mut entries: Vec<(u32, usize, f64)> = Vec::new();
        let mut deltas = vec![0.0; eliminated.width];
        for l in levels {
            entries.clear();
            for p in starts[l]..starts[l + 1] {
                (self.rows.kept).for_each_entry(p, |c, z| entries.push((c, p, w(p) * z)));
            }
            entries.sort_unstable_by_key(|&(c, ..)| c);
            for (j, delta) in deltas.iter_mut().enumerate() {
                *delta = (starts[l]..starts[l + 1]).map(|p| x(p, j) * x(p, j)).sum();
            }
            for column in entries.chunk_by(|a, b| a.0 == b.0) {
                (column.iter()).for_each(|&(_, p, u)| projection.accumulate(p, w(p) * u, &mut fit));
                projection.solve(l, &mut fit);
                let (squared, projected) = (column.iter())
                    .fold((0.0, 0.0), |(s, q), &(_, p, u)| {
                        (s + u * u, q + u * w(p) * projection.fitted(p, &fit))
                    });
                let c = column[0].0 as usize;
                acc.norms[c] += squared;
                acc.diagonal[c] += squared - projected;
                for (j, &delta) in deltas.iter().enumerate().filter(|(_, &d)| d > 0.0) {
                    let cross: f64 = column.iter().map(|&(_, p, u)| u * x(p, j)).sum();
                    acc.single_removal[j][c] += cross * cross / delta;
                }
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
        let (starts, projection) = (&self.rows.starts, &self.projection);
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
        let (starts, projection) = (&self.rows.starts, &self.projection);
        let mut fit = projection.fit();
        for l in levels {
            let rows = starts[l]..starts[l + 1];
            (rows.clone()).for_each(|p| projection.accumulate(p, w(p) * u[p], &mut fit));
            projection.solve(l, &mut fit);
            for p in rows {
                let v = w(p) * (u[p] - w(p) * projection.fitted(p, &fit));
                self.rows.kept.add_row(p, v, acc);
            }
        }
    }
}

impl<P: LevelProjection> Operator for SegmentedDesign<P> {
    fn nrows(&self) -> usize {
        self.rows.order.len()
    }
    fn ncols(&self) -> usize {
        self.n_kept
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        match self.rows.kept.n_channels {
            2 => self.apply_rows::<2>(x, y),
            _ => self.apply_rows::<0>(x, y),
        }
    }

    fn apply_adjoint(&self, u: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let scatter = |levels: Range<usize>, acc: &mut [f64]| match &self.rows.sqrt_weights {
            Some(sw) => self.scatter(u, levels, acc, |p| sw[p]),
            None => self.scatter(u, levels, acc, |_| 1.0),
        };
        if self.tasks.0.len() <= 1 || rayon::current_num_threads() == 1 {
            y.fill(0.0);
            scatter(0..self.rows.n_levels(), y);
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
        let kept = &self.rows.kept;
        self.tasks.for_each_mut(&self.rows.starts, y, |levels, y| {
            match &self.rows.sqrt_weights {
                Some(sw) => {
                    self.project_rows(levels, y, |p| sw[p], |p| sw[p] * kept.row::<WIDTH>(p, x))
                }
                None => self.project_rows(levels, y, |_| 1.0, |p| kept.row::<WIDTH>(p, x)),
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
        match &self.rows.order {
            RowOrder::Identity(_) => self.projected_rhs_rows(|p| b[p]),
            RowOrder::Permutation(rows) => self.projected_rhs_rows(|p| b[rows[p] as usize]),
        }
    }

    /// `diag(S)`: each kept column's squared norm less its projection onto the eliminated factor.
    fn column_moments(&self) -> ColumnMoments {
        let (n, width) = (self.n_kept, self.rows.eliminated.width);
        let zeros = || ColumnMoments {
            diagonal: vec![0.0; n],
            norms: vec![0.0; n],
            single_removal: vec![vec![0.0; n]; width],
        };
        let add = |a: &mut [f64], b: &[f64]| a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        let mut moments = (self.tasks.batches())
            .map(|levels| {
                let mut acc = zeros();
                match &self.rows.sqrt_weights {
                    Some(sw) => self.moments(levels, &mut acc, |p| sw[p]),
                    None => self.moments(levels, &mut acc, |_| 1.0),
                }
                acc
            })
            .reduce_with(|mut a, b| {
                add(&mut a.diagonal, &b.diagonal);
                add(&mut a.norms, &b.norms);
                (a.single_removal.iter_mut().zip(&b.single_removal)).for_each(|(a, b)| add(a, b));
                a
            })
            .unwrap_or_else(zeros);
        // A column inside the eliminated span keeps roundoff `~ε‖u‖²`, not zero.
        (moments.diagonal.iter_mut().zip(&moments.norms)).for_each(|(d, &n)| {
            *d = if *d > 1e-12 * n { *d } else { 0.0 };
        });
        moments
    }

    fn recover(&self, b: &[f64], x: &[f64], coefficients: &mut [f64]) {
        let (rows, projection) = (&*self.rows, &self.projection);
        let (n, width) = (rows.n_levels(), rows.eliminated.width);
        let original = |p: usize| match &rows.order {
            RowOrder::Identity(_) => p,
            RowOrder::Permutation(order) => order[p] as usize,
        };
        let mut by_level = vec![0.0; n * width];
        (by_level.par_chunks_mut(width).enumerate().with_min_len(256)).for_each_init(
            || projection.fit(),
            |fit, (l, out)| {
                for p in rows.starts[l]..rows.starts[l + 1] {
                    let w = rows.sqrt_weights.as_ref().map_or(1.0, |s| s[p]);
                    let v = b[original(p)] - w * rows.kept.row::<0>(p, x);
                    projection.accumulate(p, w * v, fit);
                }
                projection.solve(l, fit);
                out.copy_from_slice(projection.coefficients(fit));
            },
        );
        for (l, level) in by_level.chunks_exact(width).enumerate() {
            for (j, &value) in level.iter().enumerate() {
                coefficients[j * n + l] = value;
            }
        }
    }

    #[cfg(test)]
    fn n_tasks(&self) -> usize {
        self.tasks.0.len()
    }
}
