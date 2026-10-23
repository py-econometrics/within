use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use schwarz_precond::{lsmr, mlsmr, MlsmrOptions, Operator};

use super::{CoefficientLayout, LsmrOptions, Run, SolveResult, Solver};
use crate::domain::{row_weight, PreparedDesign};
use crate::linalg::{dot, GramBasisWorkspace};
use crate::{SolveError, WithinError};
use segmented::ReducedDesign;

#[cfg(feature = "pair-schur-prototype")]
mod pairs;
#[cfg(any(feature = "asca-prototype", feature = "pair-schur-prototype"))]
mod pcg;
mod segmented;
#[cfg(feature = "pair-schur-prototype")]
pub use pairs::PairSchurPrototype;

#[cfg(feature = "asca-prototype")]
mod asca;
#[cfg(feature = "asca-prototype")]
pub use asca::{AscaConfig, AscaPrototype};

/// Rows per parallel task, cut at an eliminated level's end so no level spans two tasks.
const ROWS_PER_TASK: usize = 16_384;

/// Observations grouped by eliminated level, and the parallel tasks over those groups.
struct LevelGroups {
    /// Grouped row `p` is original row `order[p]`.
    order: Vec<u32>,
    /// Eliminated level `e` owns grouped rows `starts[e]..starts[e + 1]`.
    starts: Vec<usize>,
    /// Eliminated levels per task.
    tasks: Vec<Range<usize>>,
}

impl LevelGroups {
    fn build(levels: &[u32], n_levels: usize) -> Self {
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
            tasks,
        }
    }

    /// Per original row, `loadings` at that row, laid out in grouped row order.
    fn gather(&self, loadings: &[&[f64]]) -> Vec<f64> {
        (self.order.par_iter())
            .flat_map_iter(|&i| loadings.iter().map(move |z| z[i as usize]))
            .collect()
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
}

struct Elimination<'a> {
    prepared: &'a PreparedDesign<'a>,
    term: usize,
    removed: Range<usize>,
    groups: Arc<LevelGroups>,
    /// The eliminated term's coefficient columns: each level's Gram block is `width × width`.
    width: usize,
    /// Per eliminated level, a generalized inverse of its Gram block, row-major.
    inverse: Vec<f64>,
}

/// A kept coefficient column: its levels' first slot among the kept coefficients, and what it multiplies.
#[derive(Clone, Copy)]
struct KeptChannel<'p> {
    start: usize,
    levels: &'p [u32],
    /// `None` is an intercept.
    loading: Option<&'p [f64]>,
}

impl<'a> Elimination<'a> {
    fn new(prepared: &'a PreparedDesign<'a>) -> Self {
        let (term, largest) = prepared
            .design
            .terms
            .iter()
            .enumerate()
            .max_by_key(|(_, t)| t.n_dofs())
            .expect("validated nonempty design");
        let removed = largest.dofs();
        let mut e = Self {
            prepared,
            term,
            removed,
            groups: Arc::new(LevelGroups::build(largest.levels(), largest.n_levels())),
            width: largest.n_columns(),
            inverse: Vec::new(),
        };
        e.inverse = e.block_inverses();
        e
    }

    /// `Wᵀ W` from pivoted Gram–Schmidt is a generalized inverse, so `A G⁻ Aᵀ` is the exact projector.
    fn block_inverses(&self) -> Vec<f64> {
        let c = self.width;
        let loadings = self.eliminated_loadings();
        let sqrt_weights = self.prepared.sqrt_weights();
        let LevelGroups { order, starts, .. } = &*self.groups;
        let mut inverse = vec![0.0; self.n_levels() * c * c];
        (inverse.par_chunks_mut(c * c).enumerate().with_min_len(256)).for_each_init(
            || GramBasisWorkspace::new(c),
            |workspace, (l, out)| {
                let basis = workspace.orthonormalize(|block| {
                    block.fill(0.0);
                    for &i in &order[starts[l]..starts[l + 1]] {
                        let (i, w) = (i as usize, row_weight(sqrt_weights, i as usize));
                        for (j, aj) in loadings.iter().enumerate() {
                            let wa = aj.map_or(w, |z| w * z[i]);
                            for (k, ak) in loadings.iter().enumerate() {
                                block[j * c + k] += ak.map_or(wa, |z| wa * z[i]);
                            }
                        }
                    }
                });
                for row in basis.rows.chunks_exact(c) {
                    for (j, &rj) in row.iter().enumerate() {
                        for (k, &rk) in row.iter().enumerate() {
                            out[j * c + k] += rj * rk;
                        }
                    }
                }
            },
        );
        inverse
    }

    fn levels(&self) -> &'a [u32] {
        self.prepared.design.terms[self.term].levels()
    }

    /// Per eliminated coefficient column, its loading; `None` is the intercept.
    fn eliminated_loadings(&self) -> Vec<Option<&'a [f64]>> {
        let prepared: &'a PreparedDesign<'a> = self.prepared;
        let term = prepared.term(self.term);
        (0..self.width).map(|c| term.loading(c)).collect()
    }

    /// Every kept coefficient column, in kept-coefficient order.
    fn kept_channels(&self) -> Vec<KeptChannel<'a>> {
        let prepared: &'a PreparedDesign<'a> = self.prepared;
        let mut start = 0;
        let mut channels = Vec::new();
        for (index, term) in prepared.terms().enumerate() {
            if index == self.term {
                continue;
            }
            for c in 0..term.term.n_columns() {
                channels.push(KeptChannel {
                    start: start + c * term.term.n_levels(),
                    levels: term.term.levels(),
                    loading: term.loading(c),
                });
            }
            start += term.term.n_dofs();
        }
        channels
    }

    /// The assembled-`S` prototypes index one intercept per kept term and invert scalar levels.
    fn require_intercepts(&self, context: &'static str) -> Result<(), SolveError> {
        if self.prepared.design.terms.iter().any(|t| t.has_slopes()) {
            return Err(SolveError::InvalidInput {
                context,
                message: "this prototype requires intercept-only factors".into(),
            });
        }
        Ok(())
    }

    fn n_levels(&self) -> usize {
        self.removed.len() / self.width
    }

    fn n_kept(&self) -> usize {
        self.prepared.design.n_dofs - self.removed.len()
    }

    fn embed(&self, reduced: &[f64], full: &mut [f64]) {
        let start = self.removed.start;
        full[..start].copy_from_slice(&reduced[..start]);
        full[self.removed.clone()].fill(0.0);
        full[self.removed.end..].copy_from_slice(&reduced[start..]);
    }

    fn restrict(&self, full: &[f64], reduced: &mut [f64]) {
        let start = self.removed.start;
        reduced[..start].copy_from_slice(&full[..start]);
        reduced[start..].copy_from_slice(&full[self.removed.end..]);
    }

    fn scale(&self, row: usize) -> f64 {
        self.prepared.sqrt_weights().map_or(1.0, |w| w[row])
    }

    /// The eliminated coefficients fitting `values`, laid out like `removed`.
    fn coefficients(&self, values: &[f64], coefficients: &mut [f64]) {
        let (n, c) = (self.n_levels(), self.width);
        let loadings = self.eliminated_loadings();
        let mut sums = vec![0.0; n * c];
        for (i, &level) in self.levels().iter().enumerate() {
            let v = self.scale(i) * values[i];
            for (sum, a) in sums[level as usize * c..][..c].iter_mut().zip(&loadings) {
                *sum += a.map_or(v, |z| z[i] * v);
            }
        }
        for (l, sums) in sums.chunks_exact(c).enumerate() {
            let inverse = &self.inverse[l * c * c..][..c * c];
            for (j, row) in inverse.chunks_exact(c).enumerate() {
                coefficients[j * n + l] = dot(row, sums);
            }
        }
    }

    #[cfg(test)]
    fn project(&self, values: &mut [f64], coefficients: &mut [f64]) {
        self.coefficients(values, coefficients);
        let n = self.n_levels();
        let loadings = self.eliminated_loadings();
        for (i, &level) in self.levels().iter().enumerate() {
            let fit: f64 = (loadings.iter().enumerate())
                .map(|(j, a)| {
                    let b = coefficients[j * n + level as usize];
                    a.map_or(b, |z| z[i] * b)
                })
                .sum();
            values[i] -= self.scale(i) * fit;
        }
    }
}

/// The serial reference [`SegmentedDesign`] is checked against.
#[cfg(test)]
struct ProjectionScratch {
    observations: Vec<f64>,
    means: Vec<f64>,
}

#[cfg(test)]
struct ProjectedDesign<'a> {
    elimination: &'a Elimination<'a>,
    scratch: Mutex<ProjectionScratch>,
}

#[cfg(test)]
impl<'a> ProjectedDesign<'a> {
    fn new(elimination: &'a Elimination<'a>) -> Self {
        Self {
            elimination,
            scratch: Mutex::new(ProjectionScratch {
                observations: vec![0.0; elimination.prepared.design.n_obs],
                means: vec![0.0; elimination.removed.len()],
            }),
        }
    }
}

#[cfg(test)]
impl Operator for ProjectedDesign<'_> {
    fn nrows(&self) -> usize {
        self.elimination.prepared.design.n_obs
    }
    fn ncols(&self) -> usize {
        self.elimination.n_kept()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let e = self.elimination;
        y.fill(0.0);
        for channel in e.kept_channels() {
            for (i, &level) in channel.levels.iter().enumerate() {
                let x = x[channel.start + level as usize];
                y[i] += channel.loading.map_or(x, |z| z[i] * x);
            }
        }
        for (i, yi) in y.iter_mut().enumerate() {
            *yi *= e.scale(i);
        }
        let mut scratch = self.scratch.lock().expect("projection scratch poisoned");
        e.project(y, &mut scratch.means);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let e = self.elimination;
        let mut scratch = self.scratch.lock().expect("projection scratch poisoned");
        let ProjectionScratch {
            observations,
            means,
        } = &mut *scratch;
        observations.copy_from_slice(x);
        e.project(observations, means);
        y.fill(0.0);
        for channel in e.kept_channels() {
            for (i, &level) in channel.levels.iter().enumerate() {
                let v = e.scale(i) * observations[i];
                y[channel.start + level as usize] += channel.loading.map_or(v, |z| z[i] * v);
            }
        }
        Ok(())
    }
}

struct RestrictedInverse<'a> {
    elimination: &'a Elimination<'a>,
    full: &'a dyn Operator,
    scratch: Mutex<(Vec<f64>, Vec<f64>)>,
}

impl Operator for RestrictedInverse<'_> {
    fn nrows(&self) -> usize {
        self.elimination.n_kept()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let mut scratch = self.scratch.lock().expect("restriction scratch poisoned");
        let (rhs, solution) = &mut *scratch;
        self.elimination.embed(x, rhs);
        self.full.apply(rhs, solution)?;
        self.elimination.restrict(solution, y);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

impl Solver<'_> {
    /// Prototype for intercepts: fixed current preconditioner, no escalation, reduced-system residual metric.
    pub fn solve_schur_prototype(
        &self,
        y: &[f64],
        options: &LsmrOptions,
    ) -> Result<SolveResult, WithinError> {
        let start = Instant::now();
        let elimination = Elimination::new(&self.prepared);
        if elimination.n_kept() == 0 {
            return self.solve(y, options);
        }
        let map = self.preconditioner().map(|full| RestrictedInverse {
            elimination: &elimination,
            full,
            scratch: Mutex::new((vec![0.0; self.n_dofs()], vec![0.0; self.n_dofs()])),
        });
        let map = map.as_ref().map(|m| m as &dyn Operator);
        let projected = ReducedDesign::build(&elimination);
        self.solve_projected(&elimination, &projected, y, options, map, start)
    }

    /// LSMR on `(I − P_e) D_kept`, `map` preconditioning the kept coefficients, then the eliminated factor recovered.
    fn solve_projected(
        &self,
        elimination: &Elimination<'_>,
        op: &ReducedDesign,
        y: &[f64],
        options: &LsmrOptions,
        map: Option<&dyn Operator>,
        start: Instant,
    ) -> Result<SolveResult, WithinError> {
        let rhs = self.prepare(y)?;
        let mut b = op.projected_rhs(rhs.b());
        let setup = start.elapsed().as_secs_f64();
        let solve_start = Instant::now();
        let mut result = match map {
            Some(map) => mlsmr(
                op,
                &b,
                map,
                options.tol,
                options.maxiter,
                MlsmrOptions {
                    local_size: options.local_size,
                    ..Default::default()
                },
            ),
            None => lsmr(op, &b, options.tol, options.maxiter, options.local_size),
        }?;
        let mut full = vec![0.0; self.n_dofs()];
        elimination.embed(&result.x, &mut full);
        rhs.op.apply(&full, &mut b)?;
        for (bi, &original) in b.iter_mut().zip(rhs.b()) {
            *bi = original - *bi;
        }
        elimination.coefficients(&b, &mut full[elimination.removed.clone()]);
        result.x = full;
        // Recovery changes the residual, including predictions at zero-weight observations.
        result.true_residual = None;
        let solution = self.finish(
            rhs,
            Run {
                result,
                solve_secs: solve_start.elapsed().as_secs_f64(),
            },
        );
        Ok(SolveResult {
            x: solution.x,
            unidentified: self.unidentified(),
            warnings: self.warnings().to_vec(),
            layout: CoefficientLayout::from_design(&self.prepared.design),
            demeaned: solution.demeaned,
            converged: solution.converged,
            iterations: solution.iterations,
            residual: solution.residual,
            time_total: start.elapsed().as_secs_f64(),
            time_setup: setup,
            time_solve: solution.time_solve,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Design, Effect, PreconditionerConfig};
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

    /// Checks the projection against the full design operator, which shares none of its indexing.
    fn assert_schur_gram(solver: &Solver<'_>) {
        let n = solver.prepared.design.n_obs;
        let e = Elimination::new(&solver.prepared);
        let op = ProjectedDesign::new(&e);
        let x: Vec<f64> = (0..op.ncols()).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let mut ax = vec![0.0; n];
        let mut aty = vec![0.0; op.ncols()];
        op.apply(&x, &mut ax).unwrap();
        op.apply_adjoint(&y, &mut aty).unwrap();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        assert!((dot(&ax, &y) - dot(&x, &aty)).abs() < 1e-10 * (1.0 + dot(&ax, &y).abs()));
        let full_op = crate::operator::DesignOperator::new(&solver.prepared);
        let mut full = vec![0.0; solver.n_dofs()];
        full_op.apply_adjoint(&ax, &mut full).unwrap();
        let norm = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm(&full[e.removed.clone()]) < 1e-10 * norm(&ax));
        let mut embedded = vec![0.0; solver.n_dofs()];
        e.embed(&x, &mut embedded);
        let mut gathered = vec![0.0; n];
        full_op.apply(&embedded, &mut gathered).unwrap();
        let mut coefficients = vec![0.0; e.removed.len()];
        e.coefficients(&gathered, &mut coefficients);
        for (target, b) in embedded[e.removed.clone()].iter_mut().zip(coefficients) {
            *target = -b;
        }
        full_op.apply(&embedded, &mut gathered).unwrap();
        let mut gram = vec![0.0; solver.n_dofs()];
        full_op.apply_adjoint(&gathered, &mut gram).unwrap();
        let mut schur = vec![0.0; op.ncols()];
        e.restrict(&gram, &mut schur);
        op.apply_adjoint(&ax, &mut aty).unwrap();
        for (a, b) in aty.iter().zip(schur) {
            assert!((a - b).abs() < 1e-10 * (1.0 + b.abs()));
        }
    }

    #[test]
    fn projected_operator_has_adjoint_and_schur_gram() {
        let categories = Array2::from_shape_fn((120, 3), |(i, j)| {
            ((i * (2 * j + 1) + i / 7) % [5, 13, 3][j]) as u32
        });
        let weights = weights(120);
        let design = Design::from_categories(categories.view()).unwrap();
        let solver = Solver::new(design, Some(&weights), PreconditionerConfig::Diagonal).unwrap();
        assert_schur_gram(&solver);
    }

    #[test]
    fn projected_slope_operator_has_adjoint_and_schur_gram() {
        let n = 600;
        let a: Vec<u32> = (0..n).map(|i| (i % 37) as u32).collect();
        let b: Vec<u32> = (0..n).map(|i| ((i * 7 + i / 11) % 13) as u32).collect();
        let c: Vec<u32> = (0..n).map(|i| (i / 37 % 5) as u32).collect();
        let z1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        // Within each `a` level, `z2` is `c` up to `1e-7`: a near alias the block inverse must survive.
        let z2: Vec<f64> = (0..n)
            .map(|i| c[i] as f64 + 1e-7 * (i as f64).cos())
            .collect();
        let z3: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos() + 2.0).collect();
        let weights = weights(n);
        for intercept in [true, false] {
            for weights in [None, Some(&weights[..])] {
                let effects = vec![
                    Effect::new(&a, intercept, [&z1[..], &z2[..]]).unwrap(),
                    Effect::new(&b, true, [&z3[..]]).unwrap(),
                    Effect::new(&c, true, []).unwrap(),
                ];
                let solver = Solver::new(effects, weights, PreconditionerConfig::Diagonal).unwrap();
                assert_schur_gram(&solver);
            }
        }
    }
}
