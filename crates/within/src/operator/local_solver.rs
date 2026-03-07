//! Approximate Cholesky-backed local solver for Schwarz subdomains.

mod approx_chol_solver;
mod block_elim_solver;
mod transforms;

use schwarz_precond::{LocalSolveError, LocalSolver};

pub(crate) use approx_chol_solver::ApproxCholSolver;
pub(crate) use block_elim_solver::{BlockElimSolver, ReducedFactor};

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
