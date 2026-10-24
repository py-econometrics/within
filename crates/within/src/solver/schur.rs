use std::time::Instant;

use schwarz_precond::{lsmr, mlsmr, MlsmrOptions, Operator};

use super::{CoefficientLayout, LsmrOptions, PreparedRhs, RhsSolution, Run, SolveResult, Solver};
use crate::domain::elimination::Elimination;
use crate::operator::reduced_design::ReducedDesign;
use crate::operator::schwarz::restricted::RestrictedInverse;
use crate::{SolveError, WithinError};

#[cfg(feature = "pair-schur-prototype")]
mod pairs;
#[cfg(feature = "pair-schur-prototype")]
pub use pairs::PairSchurPrototype;

/// One RHS: `b` in grouped row order, projected off the eliminated factor.
struct ReducedRhs<'s> {
    rhs: PreparedRhs<'s>,
    b: Vec<f64>,
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
        let map = (self.preconditioner()).map(|full| RestrictedInverse::new(&elimination, full));
        let map = map.as_ref().map(|m| m as &dyn Operator);
        let projected = ReducedDesign::build(&elimination);
        self.solve_projected(&elimination, &projected, y, options, map, start)
    }

    fn prepare_reduced<'s>(
        &'s self,
        op: &ReducedDesign,
        y: &'s [f64],
    ) -> Result<ReducedRhs<'s>, SolveError> {
        let start = Instant::now();
        let mut rhs = self.prepare(y)?;
        let b = op.projected_rhs(rhs.b());
        rhs.gather_secs = start.elapsed().as_secs_f64();
        Ok(ReducedRhs { rhs, b })
    }

    /// The eliminated factor fit to what the kept coefficients leave, then demeaned.
    fn recover(
        &self,
        elimination: &Elimination<'_>,
        reduced: ReducedRhs<'_>,
        mut run: Run,
    ) -> Result<RhsSolution, SolveError> {
        let ReducedRhs { rhs, mut b } = reduced;
        let mut full = vec![0.0; self.n_dofs()];
        elimination.embed(&run.result.x, &mut full);
        rhs.op.apply(&full, &mut b)?;
        for (bi, &original) in b.iter_mut().zip(rhs.b()) {
            *bi = original - *bi;
        }
        elimination.coefficients(&b, &mut full[elimination.removed()]);
        run.result.x = full;
        // Recovery changes the residual, including predictions at zero-weight observations.
        run.result.true_residual = None;
        Ok(self.finish(rhs, run))
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
        let reduced = self.prepare_reduced(op, y)?;
        let setup = start.elapsed().as_secs_f64();
        let run = Run::timed(|| match map {
            Some(map) => mlsmr(
                op,
                &reduced.b,
                map,
                options.tol,
                options.maxiter,
                MlsmrOptions {
                    local_size: options.local_size,
                    ..Default::default()
                },
            ),
            None => lsmr(
                op,
                &reduced.b,
                options.tol,
                options.maxiter,
                options.local_size,
            ),
        })?;
        let solution = self.recover(elimination, reduced, run)?;
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
