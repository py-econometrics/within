use std::time::Instant;

use once_cell::sync::OnceCell;
use rayon::prelude::*;
use schwarz_precond::{
    mlsmr, EscalationPolicy, LsmrResult, LsmrStopReason, MlsmrOptions, Operator, Staleness,
};

use crate::domain::elimination::Elimination;
use crate::operator::reduced_design::ReducedDesign;
use crate::operator::schwarz::pair_schur::{build_pair_schur, PairSchur, Pairs};
use crate::operator::schwarz::{DiagonalPreconditioner, SchwarzConfig};
use crate::solver::{CoefficientLayout, LsmrOptions, Pass, RhsSolution, Run, SolveResult, Solver};
use crate::{
    BatchSolveResult, BuildError, BuildWarning, LocalSolverConfig, ReductionStrategy, SolveError,
    WithinError,
};

/// How a solve picks its preconditioner on the reduced system.
enum Ladder {
    /// The pair inverse, built at construction.
    Fixed,
    /// `diag(S)` first; the pair inverse is built once a solve stalls on it.
    Adaptive {
        diagonal: DiagonalPreconditioner,
        stall: Staleness,
    },
}

/// Reduced LSMR with additive local factor-pair Schur inverses.
pub struct PairSchurPrototype<'a> {
    solver: &'a Solver<'a>,
    elimination: Elimination<'a>,
    projected: ReducedDesign,
    config: SchwarzConfig,
    pairs: Pairs,
    ladder: Ladder,
    inverse: OnceCell<PairSchur>,
    /// Elimination, projected design and whatever preconditioner construction built.
    setup_seconds: f64,
}

impl Solver<'_> {
    /// Pairs with the eliminated factor factor their Schur complement onto the other side; `dense_threshold: 0` forces approximate Cholesky.
    pub fn pair_schur_prototype(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        self.reduced_pairs(config, reduction, Pairs::All, None)
    }

    /// [`Self::pair_schur_prototype`] without the retained pairs: block Jacobi over the kept factors.
    pub fn pair_schur_block_jacobi(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        self.reduced_pairs(config, reduction, Pairs::SchurOnly, None)
    }

    /// [`Self::pair_schur_prototype`] built only once a solve stalls on `diag(S)`.
    pub fn pair_schur_adaptive(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
        stall: Staleness,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        self.reduced_pairs(config, reduction, Pairs::All, Some(stall))
    }

    fn reduced_pairs(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
        pairs: Pairs,
        stall: Option<Staleness>,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        let start = Instant::now();
        let elimination = Elimination::new(&self.prepared);
        if elimination.n_kept() == 0 {
            return Err(SolveError::InvalidInput {
                context: "pair_schur_prototype",
                message: "at least two factors are required".into(),
            }
            .into());
        }
        config.to_approx_chol()?;
        let projected = ReducedDesign::build(&elimination);
        let ladder = match stall {
            None => Ladder::Fixed,
            Some(stall) => Ladder::Adaptive {
                diagonal: DiagonalPreconditioner::from_gram_diagonal(projected.gram_diagonal())?,
                stall,
            },
        };
        let reduced = PairSchurPrototype {
            solver: self,
            elimination,
            projected,
            config: SchwarzConfig {
                local_solver: config.clone(),
                reduction,
            },
            pairs,
            ladder,
            inverse: OnceCell::new(),
            setup_seconds: 0.0,
        };
        if let Ladder::Fixed = reduced.ladder {
            reduced.pair_inverse()?;
        }
        Ok(PairSchurPrototype {
            setup_seconds: start.elapsed().as_secs_f64(),
            ..reduced
        })
    }
}

impl PairSchurPrototype<'_> {
    /// The pair inverse, built on first use.
    fn pair_inverse(&self) -> Result<&PairSchur, BuildError> {
        (self.inverse)
            .get_or_try_init(|| build_pair_schur(&self.elimination, self.pairs, &self.config))
    }

    /// Coefficients retained by the outer Schur system.
    pub fn n_reduced(&self) -> usize {
        self.elimination.n_kept()
    }
    /// Local domains whose eliminated factor has been removed from every preconditioner application.
    pub fn schur_domains(&self) -> Result<usize, WithinError> {
        Ok(self.pair_inverse()?.schur_domains)
    }
    /// Local domains containing only retained coefficients.
    pub fn retained_domains(&self) -> Result<usize, WithinError> {
        Ok(self.pair_inverse()?.retained_domains)
    }
    /// Whether a solve has built the pair inverse.
    pub fn has_escalated(&self) -> bool {
        matches!(self.ladder, Ladder::Adaptive { .. }) && self.inverse.get().is_some()
    }
    /// Seconds spent building reduced additive Schwarz.
    pub fn preconditioner_seconds(&self) -> Result<f64, WithinError> {
        Ok(self.pair_inverse()?.seconds)
    }
    /// Seconds construction took: the elimination, the projected design and any eager build.
    pub fn setup_seconds(&self) -> f64 {
        self.setup_seconds
    }
    /// Applies the reduced additive inverse for operator checks and timing experiments.
    pub fn apply_preconditioner(&self, x: &[f64], y: &mut [f64]) -> Result<(), WithinError> {
        Ok(self.pair_inverse()?.map.apply(x, y)?)
    }
    fn warnings(&self) -> Vec<BuildWarning> {
        self.inverse
            .get()
            .map_or_else(Vec::new, |b| b.warnings.clone())
    }
    /// Every RHS in one pass; an unsettled ladder then builds the pair inverse once and resumes the stalled.
    fn solve_all(
        &self,
        ys: &[&[f64]],
        lsmr: &LsmrOptions,
    ) -> Result<(Vec<RhsSolution>, f64), WithinError> {
        let (solver, op) = (self.solver, &self.projected);
        let probe = match (&self.ladder, self.inverse.get()) {
            (Ladder::Adaptive { diagonal, stall }, None) => Some((diagonal, stall)),
            _ => None,
        };
        let settled = || {
            self.inverse
                .get()
                .expect("built at construction or by the stall")
        };
        // A stall on the last permitted iteration leaves rung 2 nothing to spend, so no build.
        let stalled = |r: &LsmrResult| {
            r.stop_reason == LsmrStopReason::Escalated && r.iterations < lsmr.maxiter
        };
        let passes = (ys.par_iter())
            .map(|y| {
                let rhs = solver.prepare_reduced(op, y)?;
                let run = Run::timed(|| match probe {
                    Some((diagonal, stall)) => {
                        let options = MlsmrOptions {
                            escalation: Some(stall as &dyn EscalationPolicy),
                            local_size: lsmr.local_size,
                            ..Default::default()
                        };
                        mlsmr(op, &rhs.b, diagonal, lsmr.tol, lsmr.maxiter, options)
                    }
                    None => {
                        let options = MlsmrOptions {
                            local_size: lsmr.local_size,
                            ..Default::default()
                        };
                        let map = &settled().map;
                        mlsmr(op, &rhs.b, map, lsmr.tol, lsmr.maxiter, options)
                    }
                })?;
                Ok(match stalled(&run.result) {
                    true => Pass::Stalled(rhs, run),
                    false => Pass::Done(solver.recover(&self.elimination, rhs, run)?),
                })
            })
            .collect::<Result<Vec<_>, SolveError>>()?;
        let build_secs = match passes.iter().any(|p| matches!(p, Pass::Stalled(..))) {
            true => self.pair_inverse()?.seconds,
            false => 0.0,
        };
        let solutions = (passes.into_par_iter())
            .map(|pass| {
                let (rhs, probe) = match pass {
                    Pass::Done(solution) => return Ok(solution),
                    Pass::Stalled(rhs, probe) => (rhs, probe),
                };
                let rung1 = probe.result;
                let options = MlsmrOptions {
                    warm_start: Some(&rung1.x),
                    local_size: lsmr.local_size,
                    ..Default::default()
                };
                let remaining = lsmr.maxiter - rung1.iterations;
                let map = &settled().map;
                let mut resumed =
                    Run::timed(|| mlsmr(op, &rhs.b, map, lsmr.tol, remaining, options))?;
                resumed.result.iterations += rung1.iterations;
                resumed.solve_secs += probe.solve_secs;
                solver.recover(&self.elimination, rhs, resumed)
            })
            .collect::<Result<Vec<_>, SolveError>>()?;
        Ok((solutions, build_secs))
    }

    /// mlsmr on the projected design; `local_size` applies. Setup is the RHS projection plus any build.
    pub fn solve_lsmr(&self, y: &[f64], options: &LsmrOptions) -> Result<SolveResult, WithinError> {
        let start = Instant::now();
        let (solutions, build_secs) = self.solve_all(&[y], options)?;
        let solution = solutions
            .into_iter()
            .next()
            .expect("one RHS yields one solution");
        Ok(SolveResult {
            x: solution.x,
            unidentified: self.solver.unidentified(),
            warnings: self.warnings(),
            layout: CoefficientLayout::from_design(&self.solver.prepared.design),
            demeaned: solution.demeaned,
            converged: solution.converged,
            iterations: solution.iterations,
            residual: solution.residual,
            time_total: start.elapsed().as_secs_f64(),
            time_setup: solution.gather_secs + build_secs,
            time_solve: solution.time_solve,
        })
    }

    /// [`Self::solve_lsmr`] for every column of `ys` at once, sharing the reduced setup.
    pub fn solve_lsmr_batch(
        &self,
        ys: &[&[f64]],
        options: &LsmrOptions,
    ) -> Result<BatchSolveResult, WithinError> {
        let start = Instant::now();
        let (solutions, build_secs) = self.solve_all(ys, options)?;
        let design = &self.solver.prepared.design;
        let mut result = BatchSolveResult {
            x: Vec::with_capacity(design.n_dofs * ys.len()),
            unidentified: self.solver.unidentified(),
            warnings: self.warnings(),
            layout: CoefficientLayout::from_design(design),
            demeaned: Vec::with_capacity(design.n_obs * ys.len()),
            converged: Vec::new(),
            iterations: Vec::new(),
            residual: Vec::new(),
            time_solve: Vec::new(),
            time_setup: build_secs,
            time_total: 0.0,
            n_dofs: design.n_dofs,
            n_obs: design.n_obs,
        };
        for solution in solutions {
            result.x.extend_from_slice(&solution.x);
            result.demeaned.extend_from_slice(&solution.demeaned);
            result.converged.push(solution.converged);
            result.iterations.push(solution.iterations);
            result.residual.push(solution.residual);
            result.time_solve.push(solution.time_solve);
        }
        result.time_total = start.elapsed().as_secs_f64();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PreconditionerConfig, SchurMode};
    use ndarray::Array2;

    #[test]
    fn removed_recovery_changes_only_the_schur_nullspace_on_compatible_residuals() {
        for disconnected in [false, true] {
            let mut state = 731u64;
            let categories = Array2::from_shape_fn((1200, 3), |(i, j)| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let n = [40, 12, 6][j] / if disconnected { 2 } else { 1 };
                ((state >> 32) as usize % n + if disconnected { (i % 2) * n } else { 0 }) as u32
            });
            let config = LocalSolverConfig {
                dense_threshold: 0,
                schur: SchurMode::Exact,
                ..Default::default()
            };
            let full = Solver::new(
                categories.view(),
                None,
                PreconditionerConfig::Additive {
                    local_solver: config.clone(),
                    reduction: Default::default(),
                },
            )
            .unwrap();
            let reduced = full
                .pair_schur_prototype(&config, Default::default())
                .unwrap();
            let e = &reduced.elimination;
            let x: Vec<f64> = (0..e.n_kept()).map(|i| (i as f64).sin()).collect();
            let mut r = vec![0.0; x.len()];
            let mut scratch = vec![0.0; reduced.projected.nrows()];
            reduced.projected.apply(&x, &mut scratch).unwrap();
            reduced.projected.apply_adjoint(&scratch, &mut r).unwrap();
            let mut local = vec![0.0; x.len()];
            reduced.apply_preconditioner(&r, &mut local).unwrap();
            let mut padded = vec![0.0; full.n_dofs()];
            let mut solved = padded.clone();
            e.embed(&r, &mut padded);
            full.preconditioner()
                .unwrap()
                .apply(&padded, &mut solved)
                .unwrap();
            let mut restricted = vec![0.0; x.len()];
            e.restrict(&solved, &mut restricted);
            let delta: Vec<f64> = local.iter().zip(&restricted).map(|(a, b)| a - b).collect();
            let mut image = vec![0.0; x.len()];
            reduced.projected.apply(&delta, &mut scratch).unwrap();
            reduced
                .projected
                .apply_adjoint(&scratch, &mut image)
                .unwrap();
            let norm = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(norm(&image) / norm(&r) < 1e-10);
        }
    }

    #[test]
    fn reduced_slope_fits_match_the_full_solve() {
        use crate::solver::tests::{akm_panel, max_abs_group_mean, SlopeSpec};
        let options = LsmrOptions {
            tol: 1e-12,
            maxiter: 20_000,
            ..Default::default()
        };
        let specs = [
            SlopeSpec::Independent,
            SlopeSpec::YearIndex,
            SlopeSpec::SharedWithFirm,
            SlopeSpec::NearYearIndex(1e-10),
            SlopeSpec::NearYearIndex(1e-8),
            SlopeSpec::NearYearIndex(1e-6),
            SlopeSpec::NearYearIndex(1e-3),
            SlopeSpec::DuplicateYearIndex,
            SlopeSpec::YearIndexWithoutIntercept,
            SlopeSpec::TwoIndependentAliases,
            SlopeSpec::AgeCohort,
        ];
        let config = LocalSolverConfig::default();
        let mut escalated = false;
        for spec in specs {
            let panel = akm_panel(4_000, 200, 10, 0.15, spec);
            let full = Solver::new(
                panel.effects(),
                None,
                PreconditionerConfig::Additive {
                    local_solver: config.clone(),
                    reduction: Default::default(),
                },
            )
            .unwrap();
            let gap = |a: &[f64], b: &[f64]| {
                let scale = b.iter().map(|v| v * v).sum::<f64>().sqrt();
                (a.iter().zip(b))
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / scale
            };
            let reference = full.solve(&panel.y, &options).unwrap();
            // Near aliases leave the fit undetermined past `tol`: bound by the full solve's own spread.
            let tight = Solver::new(panel.effects(), None, PreconditionerConfig::Diagonal)
                .unwrap()
                .solve(
                    &panel.y,
                    &LsmrOptions {
                        tol: 1e-14,
                        ..options.clone()
                    },
                )
                .unwrap();
            let bound = 1e-8 + 10.0 * gap(&reference.demeaned, &tight.demeaned);
            let all = full.pair_schur_prototype(&config, Default::default());
            let jacobi = full.pair_schur_block_jacobi(&config, Default::default());
            let stall = Staleness::default();
            let adaptive = full.pair_schur_adaptive(&config, Default::default(), stall);
            for (name, reduced) in [("all", all), ("jacobi", jacobi), ("adaptive", adaptive)] {
                let reduced = reduced.unwrap();
                let doubled: Vec<f64> = panel.y.iter().map(|v| 2.0 * v).collect();
                let batch = reduced
                    .solve_lsmr_batch(&[&panel.y, &doubled], &options)
                    .unwrap();
                escalated |= reduced.has_escalated();
                for (k, scale) in [(0, 1.0), (1, 2.0)] {
                    let demeaned: Vec<f64> = batch.demeaned(k).iter().map(|v| v / scale).collect();
                    let gm = max_abs_group_mean(&full.prepared.design, &demeaned);
                    let gap = gap(&demeaned, &reference.demeaned);
                    let (converged, iterations) = (batch.converged[k], batch.iterations[k]);
                    assert!(
                        converged && gm < 1e-9 && gap < bound,
                        "{spec:?} {name}[{k}]: converged={converged} iterations={iterations} gm={gm:.2e} gap={gap:.2e} bound={bound:.2e}",
                    );
                }
            }
        }
        assert!(
            escalated,
            "no panel stalled on diag(S), so the resumed rung went untested"
        );
    }
}
