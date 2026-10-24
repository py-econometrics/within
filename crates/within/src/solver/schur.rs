use std::time::Instant;

use schwarz_precond::{Operator, Staleness};

use super::ladder::PrecondSlot;
use super::{
    batch_result, solve_all, solve_result, Formulation, LsmrOptions, PreparedRhs, RhsSolution, Run,
    SolveResult, Solver,
};
use crate::domain::elimination::{Elimination, Pairs, ReducedPairs};
use crate::linalg::dot;
use crate::operator::reduced_design::ReducedDesign;
use crate::operator::schwarz::{build_schwarz, Preconditioner, SchwarzConfig};
use crate::{
    BatchSolveResult, BuildError, BuildWarning, LocalSolverConfig, PreconditionerConfig,
    ReductionStrategy, SolveError, WithinError,
};

/// The kept coefficients' system `(I − P_e) W^{1/2} D_kept`; the eliminated term is recovered after.
struct Reduced<'a> {
    elimination: Elimination<'a>,
    design: ReducedDesign,
    pairs: Pairs,
}

/// One RHS: `b` in grouped row order, projected off the eliminated factor.
struct ReducedRhs<'s> {
    rhs: PreparedRhs<'s>,
    b: Vec<f64>,
    /// `‖b‖` before projection: the full problem's residual is the reduced one.
    full_norm: f64,
}

impl Formulation for Reduced<'_> {
    type Rhs<'s>
        = ReducedRhs<'s>
    where
        Self: 's;

    fn prepare<'s>(&'s self, y: &'s [f64]) -> Result<ReducedRhs<'s>, SolveError> {
        let start = Instant::now();
        let mut rhs = self.elimination.prepared().prepare(y)?;
        let b = self.design.projected_rhs(rhs.b());
        let full_norm = dot(rhs.b(), rhs.b()).sqrt();
        rhs.gather_secs = start.elapsed().as_secs_f64();
        Ok(ReducedRhs { rhs, b, full_norm })
    }

    fn system<'r>(&'r self, rhs: &'r ReducedRhs<'_>) -> (&'r dyn Operator, &'r [f64]) {
        (&self.design, &rhs.b)
    }

    fn residual_reference(&self, rhs: &ReducedRhs<'_>) -> Option<f64> {
        (rhs.full_norm > 0.0).then_some(rhs.full_norm)
    }

    /// The eliminated factor fit to what the kept coefficients leave, then demeaned.
    fn finish(&self, reduced: ReducedRhs<'_>, mut run: Run) -> Result<RhsSolution, SolveError> {
        let e = &self.elimination;
        let ReducedRhs { rhs, mut b, .. } = reduced;
        let mut full = vec![0.0; e.prepared().design.n_dofs];
        e.embed(&run.result.x, &mut full);
        rhs.op.apply(&full, &mut b)?;
        for (bi, &original) in b.iter_mut().zip(rhs.b()) {
            *bi = original - *bi;
        }
        e.coefficients(&b, &mut full[e.removed()]);
        run.result.x = full;
        // Recovery changes the residual, including predictions at zero-weight observations.
        run.result.true_residual = None;
        e.prepared().finish(rhs, run)
    }

    fn gram_diagonal(&self) -> Vec<f64> {
        self.design.gram_diagonal()
    }

    fn build_schwarz(
        &self,
        config: &SchwarzConfig,
    ) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
        let subdomains = ReducedPairs {
            elimination: &self.elimination,
            pairs: self.pairs,
        };
        build_schwarz(self.elimination.prepared(), config, &subdomains)
    }

    /// The eliminated term pairs with every kept one.
    fn has_factor_pairs(&self) -> bool {
        true
    }
}

/// Reduced LSMR with additive local factor-pair Schur inverses.
pub struct PairSchurPrototype<'a> {
    reduced: Reduced<'a>,
    slot: PrecondSlot,
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
        let local_solver = config.clone();
        let config = PreconditionerConfig::Additive {
            local_solver,
            reduction,
        };
        self.reduced_pairs(config, Pairs::All)
    }

    /// [`Self::pair_schur_prototype`] without the retained pairs: block Jacobi over the kept factors.
    pub fn pair_schur_block_jacobi(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        let local_solver = config.clone();
        let config = PreconditionerConfig::Additive {
            local_solver,
            reduction,
        };
        self.reduced_pairs(config, Pairs::SchurOnly)
    }

    /// [`Self::pair_schur_prototype`] built only once a solve stalls on `diag(S)`.
    pub fn pair_schur_adaptive(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
        stall: Staleness,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        let local_solver = config.clone();
        let config = PreconditionerConfig::Adaptive {
            local_solver,
            reduction,
            stall,
        };
        self.reduced_pairs(config, Pairs::All)
    }

    fn reduced_pairs(
        &self,
        config: PreconditionerConfig,
        pairs: Pairs,
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
        let design = ReducedDesign::build(&elimination);
        let reduced = Reduced {
            elimination,
            design,
            pairs,
        };
        let slot = PrecondSlot::build(&reduced, config)?;
        Ok(PairSchurPrototype {
            reduced,
            slot,
            setup_seconds: start.elapsed().as_secs_f64(),
        })
    }
}

impl PairSchurPrototype<'_> {
    fn map(&self) -> &Preconditioner {
        (self.slot.preconditioner()).expect("the eliminated term always yields a pair subdomain")
    }

    /// Coefficients retained by the outer Schur system.
    pub fn n_reduced(&self) -> usize {
        self.reduced.elimination.n_kept()
    }
    /// Subdomains of the map solves currently run on; none for the unescalated `diag(S)`.
    pub fn n_subdomains(&self) -> usize {
        self.map().n_subdomains()
    }
    /// Whether a solve has built the pair inverse.
    pub fn has_escalated(&self) -> bool {
        self.slot.has_escalated()
    }
    /// Seconds the current map's build took.
    pub fn preconditioner_seconds(&self) -> f64 {
        self.map().build_duration().as_secs_f64()
    }
    /// Seconds construction took: the elimination, the projected design and any eager build.
    pub fn setup_seconds(&self) -> f64 {
        self.setup_seconds
    }
    /// Applies the current map for operator checks and timing experiments.
    pub fn apply_preconditioner(&self, x: &[f64], y: &mut [f64]) -> Result<(), WithinError> {
        Ok(self.map().apply(x, y)?)
    }

    /// mlsmr on the projected design; `local_size` applies. Setup is the RHS projection plus any build.
    pub fn solve_lsmr(&self, y: &[f64], options: &LsmrOptions) -> Result<SolveResult, WithinError> {
        let start = Instant::now();
        let (solutions, build_secs) = solve_all(&self.reduced, &self.slot, &[y], options)?;
        let prepared = self.reduced.elimination.prepared();
        Ok(solve_result(
            prepared, &self.slot, solutions, build_secs, start,
        ))
    }

    /// [`Self::solve_lsmr`] for every column of `ys` at once, sharing the reduced setup.
    pub fn solve_lsmr_batch(
        &self,
        ys: &[&[f64]],
        options: &LsmrOptions,
    ) -> Result<BatchSolveResult, WithinError> {
        let start = Instant::now();
        let (solutions, build_secs) = solve_all(&self.reduced, &self.slot, ys, options)?;
        let prepared = self.reduced.elimination.prepared();
        Ok(batch_result(
            prepared, &self.slot, solutions, build_secs, start,
        ))
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
            let e = &reduced.reduced.elimination;
            let x: Vec<f64> = (0..e.n_kept()).map(|i| (i as f64).sin()).collect();
            let mut r = vec![0.0; x.len()];
            let mut scratch = vec![0.0; reduced.reduced.design.nrows()];
            reduced.reduced.design.apply(&x, &mut scratch).unwrap();
            reduced
                .reduced
                .design
                .apply_adjoint(&scratch, &mut r)
                .unwrap();
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
            reduced.reduced.design.apply(&delta, &mut scratch).unwrap();
            reduced
                .reduced
                .design
                .apply_adjoint(&scratch, &mut image)
                .unwrap();
            let norm = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(norm(&image) / norm(&r) < 1e-10);
        }
    }

    #[test]
    #[ignore = "multi-thread Schwarz apply is nondeterministic (#465); NearYearIndex(1e-8) flakes"]
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
