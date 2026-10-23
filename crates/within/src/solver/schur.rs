use std::ops::Range;
use std::sync::Mutex;
use std::time::Instant;

use schwarz_precond::{lsmr, mlsmr, MlsmrOptions, Operator};

use super::{CoefficientLayout, LsmrOptions, Run, SolveResult, Solver};
use crate::domain::PreparedDesign;
use crate::{SolveError, WithinError};
use segmented::SegmentedDesign;

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

struct Elimination<'a> {
    prepared: &'a PreparedDesign<'a>,
    term: usize,
    removed: Range<usize>,
    inverse: Vec<f64>,
}

impl<'a> Elimination<'a> {
    fn new(prepared: &'a PreparedDesign<'a>) -> Result<Self, SolveError> {
        if prepared.design.terms.iter().any(|t| t.has_slopes()) {
            return Err(SolveError::InvalidInput {
                context: "solve_schur_prototype",
                message: "the Schur prototype requires intercept-only factors".into(),
            });
        }
        let (term, largest) = prepared
            .design
            .terms
            .iter()
            .enumerate()
            .max_by_key(|(_, t)| t.n_dofs())
            .expect("validated nonempty design");
        let removed = largest.dofs();
        let inverse = prepared.gram_diagonal()[removed.clone()]
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
            .collect();
        Ok(Self {
            prepared,
            term,
            removed,
            inverse,
        })
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

    fn coefficients(&self, values: &[f64], means: &mut [f64]) {
        means.fill(0.0);
        for (i, &level) in self.prepared.design.terms[self.term]
            .levels()
            .iter()
            .enumerate()
        {
            means[level as usize] += self.scale(i) * values[i];
        }
        for (mean, &inverse) in means.iter_mut().zip(&self.inverse) {
            *mean *= inverse;
        }
    }

    #[cfg(test)]
    fn project(&self, values: &mut [f64], means: &mut [f64]) {
        self.coefficients(values, means);
        for (i, &level) in self.prepared.design.terms[self.term]
            .levels()
            .iter()
            .enumerate()
        {
            values[i] -= self.scale(i) * means[level as usize];
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
        let mut offset = 0;
        for (index, term) in e.prepared.design.terms.iter().enumerate() {
            if index == e.term {
                continue;
            }
            for (i, &level) in term.levels().iter().enumerate() {
                y[i] += x[offset + level as usize];
            }
            offset += term.n_dofs();
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
        let mut offset = 0;
        for (index, term) in e.prepared.design.terms.iter().enumerate() {
            if index == e.term {
                continue;
            }
            for (i, &level) in term.levels().iter().enumerate() {
                y[offset + level as usize] += e.scale(i) * observations[i];
            }
            offset += term.n_dofs();
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
        let elimination = Elimination::new(&self.prepared)?;
        if elimination.n_kept() == 0 {
            return self.solve(y, options);
        }
        let map = self.preconditioner().map(|full| RestrictedInverse {
            elimination: &elimination,
            full,
            scratch: Mutex::new((vec![0.0; self.n_dofs()], vec![0.0; self.n_dofs()])),
        });
        let map = map.as_ref().map(|m| m as &dyn Operator);
        let projected = SegmentedDesign::build(&elimination);
        self.solve_projected(&elimination, &projected, y, options, map, start)
    }

    /// LSMR on `(I − P_e) D_kept`, `map` preconditioning the kept coefficients, then the eliminated factor recovered.
    fn solve_projected(
        &self,
        elimination: &Elimination<'_>,
        op: &SegmentedDesign,
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
    use crate::{Design, PreconditionerConfig};
    use ndarray::Array2;

    #[test]
    fn projected_operator_has_adjoint_and_schur_gram() {
        let categories = Array2::from_shape_fn((120, 3), |(i, j)| {
            ((i * (2 * j + 1) + i / 7) % [5, 13, 3][j]) as u32
        });
        let weights: Vec<f64> = (0..120)
            .map(|i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    0.2 + (i % 3) as f64
                }
            })
            .collect();
        let design = Design::from_categories(categories.view()).unwrap();
        let solver = Solver::new(design, Some(&weights), PreconditionerConfig::Diagonal).unwrap();
        let e = Elimination::new(&solver.prepared).unwrap();
        let op = ProjectedDesign::new(&e);
        let x: Vec<f64> = (0..op.ncols()).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..120).map(|i| (i as f64 * 0.7).cos()).collect();
        let mut ax = vec![0.0; 120];
        let mut aty = vec![0.0; op.ncols()];
        op.apply(&x, &mut ax).unwrap();
        op.apply_adjoint(&y, &mut aty).unwrap();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        assert!((dot(&ax, &y) - dot(&x, &aty)).abs() < 1e-12);
        let full_op = crate::operator::DesignOperator::new(&solver.prepared);
        let mut embedded = vec![0.0; solver.n_dofs()];
        e.embed(&x, &mut embedded);
        let mut gathered = vec![0.0; 120];
        full_op.apply(&embedded, &mut gathered).unwrap();
        let mut means = vec![0.0; e.removed.len()];
        e.coefficients(&gathered, &mut means);
        for (target, mean) in embedded[e.removed.clone()].iter_mut().zip(means) {
            *target = -mean;
        }
        full_op.apply(&embedded, &mut gathered).unwrap();
        let mut gram = vec![0.0; solver.n_dofs()];
        full_op.apply_adjoint(&gathered, &mut gram).unwrap();
        let mut schur = vec![0.0; op.ncols()];
        e.restrict(&gram, &mut schur);
        op.apply_adjoint(&ax, &mut aty).unwrap();
        for (a, b) in aty.iter().zip(schur) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
