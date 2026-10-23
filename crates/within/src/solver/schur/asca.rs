use std::num::NonZeroUsize;
use std::time::Instant;

use faer::linalg::solvers::{ShapeCore, Solve};
use faer::sparse::{SparseColMat, Triplet};
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use schwarz_precond::Operator;

use super::pcg::{solve_reduced, Assembly, SchurMatrix};
use super::{Elimination, LsmrOptions, SolveResult, Solver};
use crate::{BuildError, Preconditioner, SolveError, WithinError};

/// Controls the experimental sum of whole-level covariance approximations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AscaConfig {
    /// Independent weighted observation pairs per level; `None` assembles every pair exactly.
    pub pairs_per_level: Option<NonZeroUsize>,
    /// Preconditioner-only diagonal shift relative to its largest diagonal.
    pub relative_shift: f64,
    /// Seed for local pair sampling.
    pub seed: u64,
}

impl Default for AscaConfig {
    fn default() -> Self {
        Self {
            pairs_per_level: NonZeroUsize::new(8),
            relative_shift: 1e-8,
            seed: 0,
        }
    }
}

struct CholeskyInverse<'a> {
    factor: &'a faer::sparse::linalg::solvers::Llt<usize, f64>,
}

impl Operator for CholeskyInverse<'_> {
    fn nrows(&self) -> usize {
        self.factor.nrows()
    }
    fn ncols(&self) -> usize {
        self.factor.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        y.copy_from_slice(x);
        self.factor.solve_in_place(faer::ColMut::from_slice_mut(y));
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

/// Reusable experimental reduced PCG solver with an assembled covariance preconditioner.
pub struct AscaPrototype<'a> {
    solver: &'a Solver<'a>,
    elimination: Elimination<'a>,
    matrix: SchurMatrix,
    factor: faer::sparse::linalg::solvers::Llt<usize, f64>,
    assembly_seconds: f64,
    factor_seconds: f64,
    preconditioner_nnz: usize,
}

fn invalid(message: impl Into<String>) -> SolveError {
    SolveError::InvalidInput {
        context: "asca_prototype",
        message: message.into(),
    }
}

impl Solver<'_> {
    /// Builds a reduced-only preconditioner independently of this solver's full-system preconditioner.
    pub fn asca_prototype(&self, config: AscaConfig) -> Result<AscaPrototype<'_>, WithinError> {
        if !config.relative_shift.is_finite() || config.relative_shift <= 0.0 {
            return Err(invalid("relative_shift must be finite and positive").into());
        }
        let start = Instant::now();
        let e = Elimination::new(&self.prepared);
        e.require_intercepts("asca_prototype")?;
        if e.n_kept() == 0 {
            return Err(invalid("ASCA requires at least two factors").into());
        }
        let kept: Vec<_> = self
            .prepared
            .design
            .terms
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != e.term)
            .map(|(_, term)| term)
            .collect();
        let mut offsets = Vec::with_capacity(kept.len());
        let mut offset = 0;
        for term in &kept {
            offsets.push(offset);
            offset += term.n_dofs();
        }
        let mut groups = vec![Vec::new(); e.removed.len()];
        let mut h = Assembly::new(e.n_kept());
        let mut c = Assembly::new(e.removed.len());
        let mut columns = vec![0; kept.len()];
        for (i, &level) in self.prepared.design.terms[e.term]
            .levels()
            .iter()
            .enumerate()
        {
            let w = e.scale(i).powi(2);
            if w == 0.0 {
                continue;
            }
            groups[level as usize].push((i, w));
            for ((column, term), &offset) in columns.iter_mut().zip(&kept).zip(&offsets) {
                *column = offset + term.levels()[i] as usize;
                c.add(level as usize, *column, w);
            }
            for &a in &columns {
                for &b in &columns {
                    h.add(a, b, w);
                }
            }
        }
        let mut approximation = Assembly::new(e.n_kept());
        let mut rng = SmallRng::seed_from_u64(config.seed);
        let mut difference = Vec::with_capacity(2 * kept.len());
        for group in &groups {
            let mass: f64 = group.iter().map(|&(_, w)| w).sum();
            if group.len() < 2 {
                continue;
            }
            let mut add_pair = |a: usize, b: usize, weight: f64| {
                difference.clear();
                for (term, &offset) in kept.iter().zip(&offsets) {
                    let ia = offset + term.levels()[a] as usize;
                    let ib = offset + term.levels()[b] as usize;
                    if ia != ib {
                        difference.push((ia, 1.0));
                        difference.push((ib, -1.0));
                    }
                }
                for &(i, vi) in &difference {
                    for &(j, vj) in &difference {
                        if i <= j {
                            approximation.add(i, j, weight * vi * vj);
                        }
                    }
                }
            };
            match config.pairs_per_level {
                None => {
                    for (j, &(a, wa)) in group.iter().enumerate() {
                        for &(b, wb) in &group[j + 1..] {
                            add_pair(a, b, (wa / mass) * wb);
                        }
                    }
                }
                Some(samples) => {
                    let mut cumulative = Vec::with_capacity(group.len());
                    let mut total = 0.0;
                    for &(_, weight) in group {
                        total += weight;
                        cumulative.push(total);
                    }
                    for _ in 0..samples.get() {
                        let draw_a = rng.random::<f64>() * total;
                        let draw_b = rng.random::<f64>() * total;
                        let a = cumulative
                            .partition_point(|&w| w <= draw_a)
                            .min(group.len() - 1);
                        let b = cumulative
                            .partition_point(|&w| w <= draw_b)
                            .min(group.len() - 1);
                        // Two independent weighted draws have covariance 2 S_i / mass.
                        add_pair(group[a].0, group[b].0, mass / (2.0 * samples.get() as f64));
                    }
                }
            }
        }
        let largest_diagonal = approximation
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| row.get(&i).copied().unwrap_or(0.0))
            .fold(0.0, f64::max);
        let shift = config.relative_shift
            * if largest_diagonal > 0.0 {
                largest_diagonal
            } else {
                1.0
            };
        for i in 0..e.n_kept() {
            approximation.add(i, i, shift);
        }
        let approximation = approximation.finish();
        let matrix = SchurMatrix {
            kept: h.finish(),
            coupling: c.finish(),
            inverse: e.inverse.clone(),
        };
        let triplets: Vec<_> = (0..approximation.indptr.len() - 1)
            .flat_map(|i| {
                approximation
                    .row(i)
                    .map(move |(j, a)| Triplet::new(i, j, a))
            })
            .collect();
        let sparse =
            SparseColMat::<usize, f64>::try_new_from_triplets(e.n_kept(), e.n_kept(), &triplets)
                .map_err(|err| BuildError::LocalSolverBuild(format!("ASCA assembly: {err:?}")))?;
        let preconditioner_nnz = triplets.len();
        drop(triplets);
        drop(approximation);
        drop(groups);
        let assembly_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let factor = sparse
            .sp_cholesky(faer::Side::Upper)
            .map_err(|err| BuildError::LocalSolverBuild(format!("ASCA Cholesky: {err:?}")))?;
        Ok(AscaPrototype {
            solver: self,
            elimination: e,
            matrix,
            factor,
            assembly_seconds,
            factor_seconds: start.elapsed().as_secs_f64(),
            preconditioner_nnz,
        })
    }
}

impl AscaPrototype<'_> {
    /// Seconds spent constructing the exact coefficient operator and approximate reduced matrix.
    pub fn assembly_seconds(&self) -> f64 {
        self.assembly_seconds
    }
    /// Seconds spent factoring the shifted reduced approximation.
    pub fn factor_seconds(&self) -> f64 {
        self.factor_seconds
    }
    /// Number of retained coefficients.
    pub fn n_reduced(&self) -> usize {
        self.elimination.n_kept()
    }
    /// Stored upper-triangle nonzeros before factorization.
    pub fn preconditioner_nnz(&self) -> usize {
        self.preconditioner_nnz
    }

    /// PCG uses `tol`/`maxiter`, ignores `local_size`, and reports the measured full normal-equation ratio.
    pub fn solve(&self, y: &[f64], options: &LsmrOptions) -> Result<SolveResult, WithinError> {
        self.solve_with_inverse(
            y,
            options,
            &CholeskyInverse {
                factor: &self.factor,
            },
        )
    }

    /// Control experiment: the same coefficient-space PCG with a restricted full-system inverse.
    pub fn solve_with_full_preconditioner(
        &self,
        y: &[f64],
        options: &LsmrOptions,
        full: &Preconditioner,
    ) -> Result<SolveResult, WithinError> {
        super::pcg::solve_with_full_preconditioner(
            self.solver,
            &self.elimination,
            &self.matrix,
            y,
            options,
            full,
        )
    }

    fn solve_with_inverse(
        &self,
        y: &[f64],
        options: &LsmrOptions,
        inverse: &dyn Operator,
    ) -> Result<SolveResult, WithinError> {
        solve_reduced(
            self.solver,
            &self.elimination,
            &self.matrix,
            y,
            options,
            inverse,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::ProjectedDesign;
    use super::*;
    use crate::PreconditionerConfig;
    use ndarray::Array2;

    #[test]
    fn coefficient_schur_matches_projected_gram_and_exact_covariance_factor() {
        let categories = Array2::from_shape_fn((120, 3), |(i, j)| {
            ((i * (j + 1) + i / 7) % [5, 13, 3][j]) as u32
        });
        let weights: Vec<f64> = (0..120)
            .map(|i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    0.3 + (i % 5) as f64
                }
            })
            .collect();
        let solver =
            Solver::new(categories.view(), Some(&weights), PreconditionerConfig::Off).unwrap();
        let config = AscaConfig {
            pairs_per_level: None,
            relative_shift: 1e-6,
            seed: 0,
        };
        let asca = solver.asca_prototype(config).unwrap();
        let e = &asca.elimination;
        let projected = ProjectedDesign::new(e);
        let n = e.n_kept();
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut ax = vec![0.0; 120];
        let mut gram = vec![0.0; n];
        projected.apply(&x, &mut ax).unwrap();
        projected.apply_adjoint(&ax, &mut gram).unwrap();
        let mut sx = vec![0.0; n];
        let mut scratch = vec![0.0; e.removed.len()];
        asca.matrix.apply(&x, &mut sx, &mut scratch);
        assert!(sx.iter().zip(&gram).all(|(a, b)| (a - b).abs() < 1e-11));
        let mut diagonal_max: f64 = 0.0;
        for j in 0..n {
            let mut basis = vec![0.0; n];
            basis[j] = 1.0;
            asca.matrix.apply(&basis, &mut gram, &mut scratch);
            diagonal_max = diagonal_max.max(gram[j]);
        }
        for (b, &xi) in sx.iter_mut().zip(&x) {
            *b += config.relative_shift * diagonal_max * xi;
        }
        asca.factor
            .solve_in_place(faer::ColMut::from_slice_mut(&mut sx));
        assert!(sx.iter().zip(x).all(|(a, b)| (a - b).abs() < 1e-8));
    }
}
