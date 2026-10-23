use std::sync::OnceLock;
use std::time::Instant;

use approx_chol::Side;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use schwarz_precond::{
    mlsmr, EscalationPolicy, LsmrResult, LsmrStopReason, MlsmrOptions, Operator,
    SchwarzPreconditioner, Staleness, SubdomainCore, SubdomainEntry,
};

use super::pcg::{solve_reduced, solve_with_full_preconditioner, SchurMatrix};
use super::segmented::ReducedDesign;
use super::Elimination;
use crate::block_solver::BlockSolver;
use crate::domain::factor_pairs::{
    channel_pairs, compute_partition_weights, pair_domains, PairMatrix,
};
use crate::domain::LocalDomain;
use crate::solver::{CoefficientLayout, LsmrOptions, Pass, RhsSolution, Run, SolveResult, Solver};
use crate::{
    BatchSolveResult, BuildError, BuildWarning, LocalSolverConfig, Preconditioner,
    ReductionStrategy, SolveError, WithinError,
};

/// Which factor pairs the reduced inverse keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pairs {
    /// Every pair: Schur complements plus the retained pairs' Grams, partition-weighted.
    All,
    /// Only the pairs split by the eliminated factor: block Jacobi on `S`.
    SchurOnly,
}

/// The eliminated factor's pairs factor their complement onto the other side; the rest their Gram, unless dropped.
fn pair_matrix(
    e: &Elimination<'_>,
    pair: crate::channel::ChannelPair,
    pairs: Pairs,
) -> Option<PairMatrix> {
    match (pair.rows.term == e.term, pair.cols.term == e.term) {
        (true, _) => Some(PairMatrix::Complement(Side::Right)),
        (_, true) => Some(PairMatrix::Complement(Side::Left)),
        _ if pairs == Pairs::SchurOnly => None,
        _ => Some(PairMatrix::Gram),
    }
}

/// `domain` over the kept coefficients: no complement or retained Gram touches the eliminated ones.
fn reduced(domain: LocalDomain, e: &Elimination<'_>) -> LocalDomain {
    let shift = e.removed.len() as u32;
    let end = e.removed.end as u32;
    let indices = (domain.core.global_indices().iter())
        .map(|&i| {
            debug_assert!(!e.removed.contains(&(i as usize)), "a kept coefficient");
            if i >= end {
                i - shift
            } else {
                i
            }
        })
        .collect();
    LocalDomain {
        core: SubdomainCore::uniform(indices),
        solver: domain.solver,
    }
}

/// The reduced additive inverse and what its build reported.
struct PairInverse {
    inverse: SchwarzPreconditioner<BlockSolver>,
    warnings: Vec<BuildWarning>,
    seconds: f64,
    schur_domains: usize,
    retained_domains: usize,
}

/// How a solve picks its preconditioner on the reduced system.
enum Ladder {
    /// The pair inverse, built at construction.
    Fixed,
    /// `diag(S)` first; the pair inverse is built once a solve stalls on it.
    Adaptive {
        diagonal: DiagonalInverse,
        stall: Staleness,
    },
}

/// `y = d ⊙ x`, a zero entry for a column `S` annihilates.
struct DiagonalInverse(Vec<f64>);

impl Operator for DiagonalInverse {
    fn nrows(&self) -> usize {
        self.0.len()
    }
    fn ncols(&self) -> usize {
        self.0.len()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        (y.par_iter_mut().zip(x).zip(&*self.0)).for_each(|((y, x), d)| *y = d * x);
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

/// Exact Schur PCG with additive local factor-pair Schur inverses.
pub struct PairSchurPrototype<'a> {
    solver: &'a Solver<'a>,
    elimination: Elimination<'a>,
    /// Assembled on first PCG use: reduced mlsmr never reads it.
    matrix: OnceLock<(SchurMatrix, f64)>,
    projected: ReducedDesign,
    config: LocalSolverConfig,
    reduction: ReductionStrategy,
    pairs: Pairs,
    ladder: Ladder,
    inverse: OnceCell<PairInverse>,
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
            Some(stall) => {
                let mut diagonal = projected.gram_diagonal();
                diagonal
                    .iter_mut()
                    .filter(|d| **d > 0.0)
                    .for_each(|d| *d = 1.0 / *d);
                Ladder::Adaptive {
                    diagonal: DiagonalInverse(diagonal),
                    stall,
                }
            }
        };
        let reduced = PairSchurPrototype {
            solver: self,
            elimination,
            matrix: OnceLock::new(),
            projected,
            config: config.clone(),
            reduction,
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
    fn build_inverse(&self) -> Result<PairInverse, BuildError> {
        let start = Instant::now();
        let e = &self.elimination;
        let approx_chol = self.config.to_approx_chol()?;
        let (_, all_pairs) = channel_pairs(&self.solver.prepared);
        let per_pair = (all_pairs.par_iter())
            .filter_map(|&pair| {
                let matrix = pair_matrix(e, pair, self.pairs)?;
                let built = pair_domains(
                    &self.solver.prepared,
                    pair,
                    approx_chol,
                    &self.config,
                    matrix,
                );
                Some(built.map(|(domains, warnings)| (matrix, domains, warnings)))
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let (mut domains, mut warnings) = (Vec::new(), Vec::new());
        let mut schur_domains = 0;
        for (matrix, pair_domains, pair_warnings) in per_pair {
            if let PairMatrix::Complement(_) = matrix {
                schur_domains += pair_domains.len();
            }
            domains.extend(pair_domains.into_iter().map(|d| reduced(d, e)));
            warnings.extend(pair_warnings);
        }
        let retained_domains = domains.len() - schur_domains;
        compute_partition_weights(&mut domains, e.n_kept());
        let entries = (domains.into_iter())
            .map(|LocalDomain { core, solver }| SubdomainEntry::try_new(core, solver))
            .collect::<Result<Vec<_>, _>>()
            .map_err(BuildError::Preconditioner)?;
        Ok(PairInverse {
            inverse: SchwarzPreconditioner::with_n_dofs(entries, e.n_kept(), self.reduction),
            warnings,
            seconds: start.elapsed().as_secs_f64(),
            schur_domains,
            retained_domains,
        })
    }

    /// The pair inverse, built on first use.
    fn pair_inverse(&self) -> Result<&PairInverse, BuildError> {
        self.inverse.get_or_try_init(|| self.build_inverse())
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
    /// Assembles `S` for the PCG solves unless already done, returning the assembly's seconds.
    pub fn assemble(&self) -> Result<f64, WithinError> {
        Ok(self.schur_matrix()?.1)
    }
    fn schur_matrix(&self) -> Result<&(SchurMatrix, f64), SolveError> {
        if let Some(built) = self.matrix.get() {
            return Ok(built);
        }
        let start = Instant::now();
        let matrix = SchurMatrix::build(&self.elimination)?;
        Ok(self
            .matrix
            .get_or_init(|| (matrix, start.elapsed().as_secs_f64())))
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
        Ok(self.pair_inverse()?.inverse.apply(x, y)?)
    }
    fn warnings(&self) -> Vec<BuildWarning> {
        self.inverse
            .get()
            .map_or_else(Vec::new, |b| b.warnings.clone())
    }
    /// PCG uses `tol`/`maxiter`, ignores `local_size`, and audits the full normal-equation residual.
    pub fn solve(&self, y: &[f64], options: &LsmrOptions) -> Result<SolveResult, WithinError> {
        let mut result = solve_reduced(
            self.solver,
            &self.elimination,
            &self.schur_matrix()?.0,
            y,
            options,
            &self.pair_inverse()?.inverse,
        )?;
        result.warnings = self.warnings();
        Ok(result)
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
                        let map = &settled().inverse;
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
                let map = &settled().inverse;
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

    /// Control experiment using the same Schur operator and PCG with the old full inverse restricted.
    pub fn solve_with_full_preconditioner(
        &self,
        y: &[f64],
        options: &LsmrOptions,
        full: &Preconditioner,
    ) -> Result<SolveResult, WithinError> {
        solve_with_full_preconditioner(
            self.solver,
            &self.elimination,
            &self.schur_matrix()?.0,
            y,
            options,
            full,
        )
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
            let mut scratch = vec![0.0; e.removed.len()];
            reduced
                .schur_matrix()
                .unwrap()
                .0
                .apply(&x, &mut r, &mut scratch);
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
            reduced
                .schur_matrix()
                .unwrap()
                .0
                .apply(&delta, &mut image, &mut scratch);
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
