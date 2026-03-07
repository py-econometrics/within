//! Approximate Cholesky-backed local solver for Schwarz subdomains.

mod block_elim_solver;

use approx_chol::Factor;
use rayon::prelude::*;
use schwarz_precond::{LocalSolveError, LocalSolver};

pub(crate) use block_elim_solver::{BlockElimSolver, ReducedFactor};

use crate::operator::csr_block::CsrBlock;

// ===========================================================================
// Transform helpers — sign-flipping, mean subtraction, back-substitution
// ===========================================================================

/// Minimum number of rows to trigger parallel back-substitution.
const PAR_BACKSUB_THRESHOLD: usize = 10_000;
const PAR_BACKSUB_CHUNK: usize = 4096;

/// Negate elements in `slice[from..]`.
#[inline]
pub(super) fn negate_block(slice: &mut [f64], from: usize) {
    for val in slice[from..].iter_mut() {
        *val = -*val;
    }
}

/// Subtract the mean of `slice[..n]` from those `n` elements.
#[inline]
pub(super) fn subtract_mean(slice: &mut [f64], n: usize) {
    let mean: f64 = slice[..n].iter().sum::<f64>() / n as f64;
    for val in slice[..n].iter_mut() {
        *val -= mean;
    }
}

/// Back-substitute for the eliminated block.
pub(super) fn backsub_block(
    sol_output: &mut [f64],
    rhs_slice: &[f64],
    cross_matrix: &CsrBlock,
    inv_diag: &[f64],
    sol_source: &[f64],
) {
    let n = sol_output.len();
    if n > PAR_BACKSUB_THRESHOLD {
        sol_output
            .par_chunks_mut(PAR_BACKSUB_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * PAR_BACKSUB_CHUNK;
                for (local_i, si) in chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let start = cross_matrix.indptr[i] as usize;
                    let end = cross_matrix.indptr[i + 1] as usize;
                    let mut sum = 0.0;
                    for idx in start..end {
                        let j = cross_matrix.indices[idx] as usize;
                        sum += cross_matrix.data[idx] * sol_source[j];
                    }
                    *si = inv_diag[i] * (rhs_slice[i] + sum);
                }
            });
    } else {
        for i in 0..n {
            let start = cross_matrix.indptr[i] as usize;
            let end = cross_matrix.indptr[i + 1] as usize;
            let mut sum = 0.0;
            for idx in start..end {
                let j = cross_matrix.indices[idx] as usize;
                sum += cross_matrix.data[idx] * sol_source[j];
            }
            sol_output[i] = inv_diag[i] * (rhs_slice[i] + sum);
        }
    }
}

// ===========================================================================
// ApproxCholSolver — local solver backed by approximate Cholesky
// ===========================================================================

/// Local subdomain solver backed by approximate Cholesky factorization.
pub struct ApproxCholSolver {
    factor: Factor,
    strategy: LocalSolveStrategy,
    n_local: usize,
}

impl ApproxCholSolver {
    pub(crate) fn new(factor: Factor, strategy: LocalSolveStrategy, n_local: usize) -> Self {
        Self {
            factor,
            strategy,
            n_local,
        }
    }
}

impl LocalSolver for ApproxCholSolver {
    fn n_local(&self) -> usize {
        self.n_local
    }

    fn scratch_size(&self) -> usize {
        self.factor.n()
    }

    fn solve_local(&self, rhs: &mut [f64], sol: &mut [f64]) -> Result<(), LocalSolveError> {
        let n_local = self.n_local;
        let (solve_n, fbs) = match &self.strategy {
            LocalSolveStrategy::Laplacian => {
                debug_assert_eq!(self.factor.n(), n_local);
                sol[..n_local].copy_from_slice(&rhs[..n_local]);
                self.factor
                    .solve_in_place(&mut sol[..n_local])
                    .map_err(|e| LocalSolveError::ApproxCholSolveFailed {
                        context: "within.local.approx_chol.laplacian",
                        message: e.to_string(),
                    })?;
                return Ok(());
            }
            LocalSolveStrategy::LaplacianGramian { first_block_size } => {
                (n_local, Some(*first_block_size))
            }
            LocalSolveStrategy::GramianAugmented { first_block_size } => {
                (n_local + 1, Some(*first_block_size))
            }
            LocalSolveStrategy::Sddm => (n_local + 1, None),
        };

        if let Some(fbs) = fbs {
            negate_block(&mut rhs[..n_local], fbs);
        }
        if solve_n > n_local {
            debug_assert_eq!(self.factor.n(), solve_n);
            rhs[n_local] = 0.0;
        }
        subtract_mean(rhs, solve_n);

        sol[..solve_n].copy_from_slice(&rhs[..solve_n]);
        debug_assert_eq!(self.factor.n(), solve_n);
        self.factor
            .solve_in_place(&mut sol[..solve_n])
            .map_err(|e| LocalSolveError::ApproxCholSolveFailed {
                context: "within.local.approx_chol.augmented_or_gramian",
                message: e.to_string(),
            })?;

        if let Some(fbs) = fbs {
            negate_block(&mut sol[..n_local], fbs);
        }
        Ok(())
    }
}

// ===========================================================================
// FeLocalSolver — dispatch enum
// ===========================================================================

/// FE-specific local solver, dispatching to either full-SDDM ApproxChol or
/// Schur complement block elimination.
pub enum FeLocalSolver {
    /// Full bipartite SDDM factorized via approximate Cholesky.
    FullSddm { solver: ApproxCholSolver },
    /// Schur complement reduction + reduced-system factor solve
    /// (dense anchored for tiny Schur when enabled, otherwise sparse ApproxChol).
    SchurComplement(Box<BlockElimSolver>),
}

impl LocalSolver for FeLocalSolver {
    fn n_local(&self) -> usize {
        match self {
            Self::FullSddm { solver, .. } => solver.n_local(),
            Self::SchurComplement(s) => s.n_local(),
        }
    }

    fn scratch_size(&self) -> usize {
        match self {
            Self::FullSddm { solver, .. } => solver.scratch_size(),
            Self::SchurComplement(s) => s.scratch_size(),
        }
    }

    fn solve_local(&self, rhs: &mut [f64], sol: &mut [f64]) -> Result<(), LocalSolveError> {
        match self {
            Self::FullSddm { solver, .. } => solver.solve_local(rhs, sol),
            Self::SchurComplement(s) => s.solve_local(rhs, sol),
        }
    }
}

/// Determines how the local Gramian solve is performed for a subdomain.
#[derive(Debug, Clone)]
pub enum LocalSolveStrategy {
    /// The local Gramian is naturally a graph Laplacian (no augmentation needed).
    Laplacian,

    /// The local Gramian is SDDM but not Laplacian, so Gremban augmentation added
    /// an extra node.
    Sddm,

    /// Factor-pair domain where the bipartite Gramian maps to a Laplacian via
    /// sign-flipping the second block. No augmentation was needed.
    LaplacianGramian { first_block_size: usize },

    /// Factor-pair domain where the Gramian needed Gremban augmentation.
    GramianAugmented { first_block_size: usize },
}

impl LocalSolveStrategy {
    /// Map a bipartite hint and augmentation flag to the appropriate local
    /// solve strategy.
    pub fn from_flags(first_block_size: Option<usize>, was_augmented: bool) -> Self {
        match first_block_size {
            Some(fbs) => {
                if was_augmented {
                    Self::GramianAugmented {
                        first_block_size: fbs,
                    }
                } else {
                    Self::LaplacianGramian {
                        first_block_size: fbs,
                    }
                }
            }
            None => {
                if was_augmented {
                    Self::Sddm
                } else {
                    Self::Laplacian
                }
            }
        }
    }
}
