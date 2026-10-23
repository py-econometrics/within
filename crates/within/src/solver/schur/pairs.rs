use std::sync::OnceLock;
use std::time::Instant;

use approx_chol::Side;
use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainCore, SubdomainEntry};

use super::pcg::{solve_reduced, solve_with_full_preconditioner, SchurMatrix};
use super::segmented::ReducedDesign;
use super::{Elimination, LsmrOptions, SolveResult, Solver};
use crate::block_solver::BlockSolver;
use crate::domain::factor_pairs::{
    channel_pairs, compute_partition_weights, pair_domains, PairMatrix,
};
use crate::domain::LocalDomain;
use crate::{
    BuildError, BuildWarning, LocalSolverConfig, Preconditioner, ReductionStrategy, SolveError,
    WithinError,
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

/// Exact Schur PCG with additive local factor-pair Schur inverses.
pub struct PairSchurPrototype<'a> {
    solver: &'a Solver<'a>,
    elimination: Elimination<'a>,
    /// Assembled on first PCG use: reduced mlsmr never reads it.
    matrix: OnceLock<(SchurMatrix, f64)>,
    projected: ReducedDesign,
    inverse: SchwarzPreconditioner<BlockSolver>,
    warnings: Vec<BuildWarning>,
    preconditioner_seconds: f64,
    schur_domains: usize,
    retained_domains: usize,
}

impl Solver<'_> {
    /// Pairs with the eliminated factor factor their Schur complement onto the other side; `dense_threshold: 0` forces approximate Cholesky.
    pub fn pair_schur_prototype(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        self.reduced_pairs(config, reduction, Pairs::All)
    }

    /// [`Self::pair_schur_prototype`] without the retained pairs: block Jacobi over the kept factors.
    pub fn pair_schur_block_jacobi(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        self.reduced_pairs(config, reduction, Pairs::SchurOnly)
    }

    fn reduced_pairs(
        &self,
        config: &LocalSolverConfig,
        reduction: ReductionStrategy,
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
        let projected = ReducedDesign::build(&elimination);
        let approx_chol = config.to_approx_chol()?;
        let (_, all_pairs) = channel_pairs(&self.prepared);
        let per_pair = (all_pairs.par_iter())
            .filter_map(|&pair| {
                let matrix = pair_matrix(&elimination, pair, pairs)?;
                let built = pair_domains(&self.prepared, pair, approx_chol, config, matrix);
                Some(built.map(|(domains, warnings)| (matrix, domains, warnings)))
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let (mut domains, mut warnings) = (Vec::new(), Vec::new());
        let mut schur_domains = 0;
        for (matrix, pair_domains, pair_warnings) in per_pair {
            if let PairMatrix::Complement(_) = matrix {
                schur_domains += pair_domains.len();
            }
            domains.extend(pair_domains.into_iter().map(|d| reduced(d, &elimination)));
            warnings.extend(pair_warnings);
        }
        let retained_domains = domains.len() - schur_domains;
        compute_partition_weights(&mut domains, elimination.n_kept());
        let entries = (domains.into_iter())
            .map(|LocalDomain { core, solver }| SubdomainEntry::try_new(core, solver))
            .collect::<Result<Vec<_>, _>>()
            .map_err(BuildError::Preconditioner)?;
        let inverse = SchwarzPreconditioner::with_n_dofs(entries, elimination.n_kept(), reduction);
        Ok(PairSchurPrototype {
            solver: self,
            elimination,
            matrix: OnceLock::new(),
            projected,
            inverse,
            warnings,
            preconditioner_seconds: start.elapsed().as_secs_f64(),
            schur_domains,
            retained_domains,
        })
    }
}

impl PairSchurPrototype<'_> {
    /// Coefficients retained by the outer Schur system.
    pub fn n_reduced(&self) -> usize {
        self.elimination.n_kept()
    }
    /// Local domains whose eliminated factor has been removed from every preconditioner application.
    pub fn schur_domains(&self) -> usize {
        self.schur_domains
    }
    /// Local domains containing only retained coefficients.
    pub fn retained_domains(&self) -> usize {
        self.retained_domains
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
    pub fn preconditioner_seconds(&self) -> f64 {
        self.preconditioner_seconds
    }
    /// Applies the reduced additive inverse for operator checks and timing experiments.
    pub fn apply_preconditioner(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.inverse.apply(x, y)
    }
    /// PCG uses `tol`/`maxiter`, ignores `local_size`, and audits the full normal-equation residual.
    pub fn solve(&self, y: &[f64], options: &LsmrOptions) -> Result<SolveResult, WithinError> {
        let mut result = solve_reduced(
            self.solver,
            &self.elimination,
            &self.schur_matrix()?.0,
            y,
            options,
            &self.inverse,
        )?;
        result.warnings = self.warnings.clone();
        Ok(result)
    }
    /// mlsmr on the projected design with the same reduced additive inverse; `local_size` applies.
    pub fn solve_lsmr(&self, y: &[f64], options: &LsmrOptions) -> Result<SolveResult, WithinError> {
        let map: &dyn Operator = &self.inverse;
        let mut result = (self.solver).solve_projected(
            &self.elimination,
            &self.projected,
            y,
            options,
            Some(map),
            Instant::now(),
        )?;
        result.warnings = self.warnings.clone();
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
            for (name, reduced) in [("all", all), ("jacobi", jacobi)] {
                let out = reduced.unwrap().solve_lsmr(&panel.y, &options).unwrap();
                let gm = max_abs_group_mean(&full.prepared.design, &out.demeaned);
                let gap = gap(&out.demeaned, &reference.demeaned);
                assert!(
                    out.converged && gm < 1e-9 && gap < bound,
                    "{spec:?} {name}: converged={} iterations={} gm={gm:.2e} gap={gap:.2e} bound={bound:.2e}",
                    out.converged,
                    out.iterations
                );
            }
        }
    }
}
