use std::sync::Arc;
use std::time::Instant;

use schwarz_precond::{Operator, Staleness};

use super::ladder::{CoarseSource, PrecondSlot};
use super::{
    batch_result, solve_all, solve_result, Formulation, LsmrOptions, PreparedRhs, RhsSolution, Run,
    SolveResult, Solver,
};
use approx_chol::Side;

use crate::channel::ChannelPair;
use crate::domain::elimination::Elimination;
use crate::domain::factor_pairs::PairMatrix;
use crate::domain::{PairSubdomains, PreparedDesign};
use crate::linalg::dot;
use crate::operator::coarse::{CoarseRung, Coarsened};
use crate::operator::reduced_coarse::{Declined, ReducedCoarse};
use crate::operator::reduced_design::{self, ColumnMoments, ReducedDesign};
use crate::operator::schwarz::{build_schwarz, Preconditioner};
use crate::{
    BatchSolveResult, BuildError, BuildWarning, LocalSolverConfig, PreconditionerConfig,
    SolveError, WithinError,
};

/// The kept coefficients' system `(I − P_e) W^{1/2} D_kept`; the eliminated term is recovered after.
struct Reduced<'a> {
    elimination: Elimination<'a>,
    design: Box<dyn ReducedDesign>,
    pairs: Pairs,
    /// Set by `WITHIN_REDUCED_GAMMA` (Gram pieces) and `WITHIN_REDUCED_GAMMA_C` (complement pieces).
    scales: Option<PieceScales>,
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
        (self.design.operator(), &rhs.b)
    }

    fn block_design(&self) -> Option<&dyn crate::operator::reduced_design::BlockDesign> {
        Some(self.design.block())
    }

    fn residual_reference(&self, rhs: &ReducedRhs<'_>) -> Option<f64> {
        (rhs.full_norm > 0.0).then_some(rhs.full_norm)
    }

    /// The eliminated factor fit to what the kept coefficients leave, then demeaned.
    fn finish(&self, reduced: ReducedRhs<'_>, mut run: Run) -> Result<RhsSolution, SolveError> {
        let e = &self.elimination;
        let rhs = reduced.rhs;
        let mut full = vec![0.0; e.prepared().design.n_dofs];
        e.embed(&run.result.x, &mut full);
        (self.design).recover(rhs.b(), &run.result.x, &mut full[e.removed()]);
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
        config: &LocalSolverConfig,
    ) -> Result<(Option<Preconditioner>, Vec<BuildWarning>), BuildError> {
        let subdomains = ReducedPairs {
            elimination: &self.elimination,
            pairs: self.pairs,
            scales: self.scales.as_ref(),
        };
        build_schwarz(self.elimination.prepared(), config, &subdomains)
    }

    /// The eliminated term pairs with every kept one.
    fn has_factor_pairs(&self) -> bool {
        true
    }

    fn coarse_source(&self) -> Option<&dyn CoarseSource> {
        Some(self)
    }

    /// `WITHIN_REDUCED_RIGID` builds the coarse rung with a fixed map too.
    fn eager_coarse(&self) -> Option<&dyn CoarseSource> {
        std::env::var_os("WITHIN_REDUCED_RIGID").map(|_| self as &dyn CoarseSource)
    }
}

impl CoarseSource for Reduced<'_> {
    /// Gluing more spells coarsens the space and thins the factor: double `K` until it fits.
    /// Past `8 K` gluing saturates and `κ ~ K` erodes what the rung buys.
    fn build_coarse(&self) -> Option<Box<dyn CoarseRung>> {
        let first = rigid_k();
        let (mut k, mut glued) = (first, None);
        loop {
            match ReducedCoarse::build(&self.elimination, k) {
                Ok(coarse) => return Some(Box::new(coarse)),
                Err(Declined::OverBudget { glued: now })
                    if k < 8.0 * first && glued != Some(now) =>
                {
                    (k, glued) = (2.0 * k, Some(now));
                }
                Err(_) => return None,
            }
        }
    }
}

/// Initial spell-gluing ratio `K`, `WITHIN_REDUCED_RIGID` overriding 20.
fn rigid_k() -> f64 {
    (std::env::var("WITHIN_REDUCED_RIGID").ok())
        .and_then(|k| k.parse::<f64>().ok())
        .unwrap_or(20.0)
}

/// Which factor pairs the reduced subdomains keep.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pairs {
    /// Every pair: Schur complements plus the retained pairs' Grams, partition-weighted.
    All,
    /// Only the pairs split by the eliminated factor: block Jacobi on `S`.
    SchurOnly,
}

/// Subdomains over the kept coefficients: the eliminated term's pairs factor their complement.
struct ReducedPairs<'e> {
    elimination: &'e Elimination<'e>,
    pairs: Pairs,
    scales: Option<&'e PieceScales>,
}

/// Per kept coefficient, `(S_ll / piece_ll)^{-γ}`: one vector for the retained Grams, one per eliminated column.
struct PieceScales {
    gram: Arc<Vec<f64>>,
    complement: Vec<Arc<Vec<f64>>>,
}

impl PieceScales {
    /// `γ = 0` keeps a piece's plain weight; so does a coefficient inside the eliminated span.
    fn new(moments: &ColumnMoments, gamma_gram: f64, gamma_complement: f64) -> Self {
        let scale = |piece: &[f64], g: f64| -> Arc<Vec<f64>> {
            Arc::new(
                (moments.diagonal.iter().zip(piece))
                    .map(|(&s, &p)| {
                        if s > 0.0 && p > s * 1e-12 && g != 0.0 {
                            (s / p).powf(-g)
                        } else {
                            1.0
                        }
                    })
                    .collect(),
            )
        };
        let complement = (moments.single_removal.iter())
            .map(|removed| {
                let piece: Vec<f64> = (moments.norms.iter().zip(removed))
                    .map(|(a, r)| a - r)
                    .collect();
                scale(&piece, gamma_complement)
            })
            .collect();
        Self {
            gram: scale(&moments.norms, gamma_gram),
            complement,
        }
    }
}

impl PairSubdomains for ReducedPairs<'_> {
    fn matrix(&self, pair: ChannelPair) -> Option<PairMatrix> {
        let term = self.elimination.term();
        match (pair.rows.term == term, pair.cols.term == term) {
            (true, _) => Some(PairMatrix::Complement(Side::Right)),
            (_, true) => Some(PairMatrix::Complement(Side::Left)),
            _ if self.pairs == Pairs::SchurOnly => None,
            _ => Some(PairMatrix::Gram),
        }
    }

    fn dof(&self, dof: usize) -> usize {
        let removed = self.elimination.removed();
        debug_assert!(!removed.contains(&dof), "a kept coefficient");
        if dof >= removed.end {
            dof - removed.len()
        } else {
            dof
        }
    }

    fn n_dofs(&self, _: &PreparedDesign<'_>) -> usize {
        self.elimination.n_kept()
    }

    /// Unlike the full design's, slope overlaps do not slow the reduced solve under `1/√c`.
    fn partition_weighted(&self, _: &PreparedDesign<'_>) -> bool {
        true
    }

    fn piece_scale(&self, pair: ChannelPair) -> Option<Arc<Vec<f64>>> {
        let (scales, term) = (self.scales?, self.elimination.term());
        Some(match (pair.rows.term == term, pair.cols.term == term) {
            (true, _) => scales.complement[pair.rows.column].clone(),
            (_, true) => scales.complement[pair.cols.column].clone(),
            _ => scales.gram.clone(),
        })
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
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        let local_solver = config.clone();
        let config = PreconditionerConfig::Additive {
            local_solver,
        };
        self.reduced_pairs(config, Pairs::All)
    }

    /// [`Self::pair_schur_prototype`] without the retained pairs: block Jacobi over the kept factors.
    pub fn pair_schur_block_jacobi(
        &self,
        config: &LocalSolverConfig,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        let local_solver = config.clone();
        let config = PreconditionerConfig::Additive {
            local_solver,
        };
        self.reduced_pairs(config, Pairs::SchurOnly)
    }

    /// [`Self::pair_schur_prototype`] built only once a solve stalls on `diag(S)`.
    pub fn pair_schur_adaptive(
        &self,
        config: &LocalSolverConfig,
        stall: Staleness,
    ) -> Result<PairSchurPrototype<'_>, WithinError> {
        let local_solver = config.clone();
        let config = PreconditionerConfig::Adaptive {
            local_solver,
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
        let design = reduced_design::build(&elimination);
        let gamma = |name: &str| {
            (std::env::var(name).ok())
                .and_then(|g| g.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        let (gamma_g, gamma_c) = (
            gamma("WITHIN_REDUCED_GAMMA"),
            gamma("WITHIN_REDUCED_GAMMA_C"),
        );
        let scales = (gamma_g != 0.0 || gamma_c != 0.0)
            .then(|| PieceScales::new(&design.column_moments(), gamma_g, gamma_c));
        let reduced = Reduced {
            elimination,
            design,
            pairs,
            scales,
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

    /// Ritz pairs `(θ, residual bound)` of `M S` (`M` the map with its coarse rung) via Lanczos on `Ã M Ãᵀ`.
    pub fn ritz_values(&self, steps: usize) -> Result<Vec<(f64, f64)>, WithinError> {
        let (diag, off) = self.with_map(|m| self.lanczos(m, steps, |_, _| {}))?;
        let (theta, u) = tridiagonal_eigen(&diag, &off);
        let (k, tail) = (diag.len(), off.get(diag.len() - 1).copied().unwrap_or(0.0));
        Ok((0..k)
            .map(|j| (theta[j], tail * u[(k - 1, j)].abs()))
            .collect())
    }

    /// Eigenvectors `x = M Ãᵀ u` of `M S` for the Ritz indices `which` of [`Self::ritz_values`],
    /// replayed, with `‖M S x − θ x‖ / ‖θ x‖`.
    pub fn ritz_vectors(
        &self,
        steps: usize,
        which: &[usize],
    ) -> Result<Vec<(Vec<f64>, f64)>, WithinError> {
        self.with_map(|m| {
            let (diag, off) = self.lanczos(m, steps, |_, _| {})?;
            let (theta, coef) = tridiagonal_eigen(&diag, &off);
            let rows = self.reduced.design.nrows();
            let mut u = vec![vec![0.0; rows]; which.len()];
            self.lanczos(m, diag.len(), |j, v| {
                for (u, &i) in u.iter_mut().zip(which) {
                    let c = coef[(j, i)];
                    u.iter_mut().zip(v).for_each(|(u, v)| *u += c * v);
                }
            })?;
            let (n, design) = (self.n_reduced(), &self.reduced.design);
            let (mut g, mut r, mut y) = (vec![0.0; n], vec![0.0; rows], vec![0.0; n]);
            (u.iter().zip(which))
                .map(|(u, &i)| {
                    let mut x = vec![0.0; n];
                    design.apply_adjoint(u, &mut g)?;
                    m.apply(&g, &mut x)?;
                    design.apply(&x, &mut r)?;
                    design.apply_adjoint(&r, &mut g)?;
                    m.apply(&g, &mut y)?;
                    y.iter_mut().zip(&x).for_each(|(y, x)| *y -= theta[i] * x);
                    let residual = (dot(&y, &y) / dot(&x, &x)).sqrt() / theta[i];
                    Ok((x, residual))
                })
                .collect()
        })
    }

    /// `(levels, columns)` of each kept term, in the order of the reduced coefficients.
    pub fn kept_terms(&self) -> Vec<(usize, usize)> {
        let e = &self.reduced.elimination;
        (e.prepared().design.terms.iter().enumerate())
            .filter(|&(t, _)| t != e.term())
            .map(|(_, term)| (term.n_levels(), term.n_columns()))
            .collect()
    }

    fn with_map<R>(&self, f: impl FnOnce(&dyn Operator) -> R) -> R {
        let PrecondSlot::Static {
            map: Some(fixed), ..
        } = &self.slot
        else {
            panic!("Ritz pairs need a fixed map");
        };
        match fixed.coarse.as_deref() {
            Some(coarse) => f(&Coarsened {
                inner: &fixed.inner,
                coarse,
            }),
            None => f(&fixed.inner),
        }
    }

    /// Plain Lanczos on `Ã M Ãᵀ` from a fixed start; `visit(j, v_j)` sees each basis vector.
    fn lanczos(
        &self,
        m: &dyn Operator,
        steps: usize,
        mut visit: impl FnMut(usize, &[f64]),
    ) -> Result<(Vec<f64>, Vec<f64>), WithinError> {
        let (n, design) = (self.n_reduced(), &self.reduced.design);
        // `S`'s gauge null space never enters: every product ends in `range(Ã)`.
        let rows = design.nrows();
        let (mut g, mut mg) = (vec![0.0; n], vec![0.0; n]);
        let x0: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut v = vec![0.0; rows];
        design.apply(&x0, &mut v)?;
        let norm = dot(&v, &v).sqrt();
        v.iter_mut().for_each(|a| *a /= norm);
        let (mut v_prev, mut w) = (vec![0.0; rows], vec![0.0; rows]);
        let (mut diag, mut off) = (Vec::new(), Vec::new());
        let mut beta = 0.0;
        for j in 0..steps {
            visit(j, &v);
            design.apply_adjoint(&v, &mut g)?;
            m.apply(&g, &mut mg)?;
            design.apply(&mg, &mut w)?;
            w.iter_mut().zip(&v_prev).for_each(|(w, p)| *w -= beta * p);
            let alpha = dot(&v, &w);
            w.iter_mut().zip(&v).for_each(|(w, v)| *w -= alpha * v);
            diag.push(alpha);
            beta = dot(&w, &w).sqrt();
            if beta.partial_cmp(&(1e-12 * alpha.abs())) != Some(std::cmp::Ordering::Greater) {
                break;
            }
            off.push(beta);
            std::mem::swap(&mut v_prev, &mut v);
            v.iter_mut().zip(&w).for_each(|(v, w)| *v = w / beta);
        }
        Ok((diag, off))
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

fn tridiagonal_eigen(diag: &[f64], off: &[f64]) -> (Vec<f64>, faer::Mat<f64>) {
    let k = diag.len();
    let t = faer::Mat::<f64>::from_fn(k, k, |i, j| match i.abs_diff(j) {
        0 => diag[i],
        1 => off[i.min(j)],
        _ => 0.0,
    });
    let evd = (t.self_adjoint_eigen(faer::Side::Lower)).expect("tridiagonal eigenvalues");
    let theta = (0..k).map(|j| evd.S()[j]).collect();
    (theta, evd.U().to_owned())
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
                },
            )
            .unwrap();
            let reduced = full
                .pair_schur_prototype(&config)
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
            let all = full.pair_schur_prototype(&config);
            let jacobi = full.pair_schur_block_jacobi(&config);
            let stall = Staleness::default();
            let adaptive = full.pair_schur_adaptive(&config, stall);
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
