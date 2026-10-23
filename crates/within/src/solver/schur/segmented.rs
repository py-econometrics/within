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
    inverse: Vec<f64>,
    /// Each grouped row's reduced coefficient per kept term, row-major.
    columns: Vec<u32>,
    n_terms: usize,
    n_kept: usize,
    sqrt_weights: Option<Vec<f64>>,
    /// Eliminated levels per task.
    tasks: Vec<Range<usize>>,
}

impl SegmentedDesign {
    pub(super) fn build(e: &Elimination<'_>) -> Self {
        let design = &e.prepared.design;
        let levels = design.terms[e.term].levels();
        let n_levels = e.removed.len();
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
        let mut offset = 0u32;
        let kept: Vec<(u32, &[u32])> = (design.terms.iter().enumerate())
            .filter(|&(t, _)| t != e.term)
            .map(|(_, term)| {
                let column = (offset, term.levels());
                offset += term.n_dofs() as u32;
                column
            })
            .collect();
        let columns = order
            .par_iter()
            .flat_map_iter(|&i| kept.iter().map(move |&(at, l)| at + l[i as usize]))
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
            inverse: e.inverse.clone(),
            columns,
            n_terms: kept.len(),
            n_kept: offset as usize,
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

    fn columns(&self, p: usize) -> &[u32] {
        &self.columns[p * self.n_terms..][..self.n_terms]
    }

    /// `y ← (I − P_e) y` on `levels`' rows, `y` starting at the first of them.
    #[inline(always)]
    fn center(&self, levels: Range<usize>, y: &mut [f64], w: impl Fn(usize) -> f64) {
        let base = self.starts[levels.start];
        for l in levels {
            let rows = self.starts[l]..self.starts[l + 1];
            let sum: f64 = rows.clone().map(|p| w(p) * y[p - base]).sum();
            let mean = sum * self.inverse[l];
            rows.for_each(|p| y[p - base] -= w(p) * mean);
        }
    }

    #[inline(always)]
    fn gather(&self, x: &[f64], levels: Range<usize>, y: &mut [f64], w: impl Fn(usize) -> f64) {
        let base = self.starts[levels.start];
        for l in levels {
            let rows = self.starts[l]..self.starts[l + 1];
            let mut sum = 0.0;
            for p in rows.clone() {
                let v = w(p) * self.columns(p).iter().map(|&c| x[c as usize]).sum::<f64>();
                y[p - base] = v;
                sum += w(p) * v;
            }
            let mean = sum * self.inverse[l];
            rows.for_each(|p| y[p - base] -= w(p) * mean);
        }
    }

    #[inline(always)]
    fn scatter(&self, u: &[f64], levels: Range<usize>, acc: &mut [f64], w: impl Fn(usize) -> f64) {
        for l in levels {
            let rows = self.starts[l]..self.starts[l + 1];
            let sum: f64 = rows.clone().map(|p| w(p) * u[p]).sum();
            let mean = sum * self.inverse[l];
            for p in rows {
                let v = w(p) * (u[p] - w(p) * mean);
                self.columns(p).iter().for_each(|&c| acc[c as usize] += v);
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
    use crate::{Design, PreconditionerConfig, Solver};
    use ndarray::Array2;

    #[test]
    fn grouped_rows_reproduce_the_projected_design() {
        let n = 30_000;
        let categories = Array2::from_shape_fn((n, 3), |(i, j)| {
            ((i * (2 * j + 1) + i / 7) % [5003, 131, 7][j]) as u32
        });
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    0.2 + (i % 3) as f64
                }
            })
            .collect();
        for weights in [None, Some(&weights[..])] {
            let design = Design::from_categories(categories.view()).unwrap();
            let solver = Solver::new(design, weights, PreconditionerConfig::Diagonal).unwrap();
            let e = Elimination::new(&solver.prepared).unwrap();
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
        }
    }
}
