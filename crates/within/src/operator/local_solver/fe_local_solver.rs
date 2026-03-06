use schwarz_precond::{LocalSolveError, LocalSolver};

use super::approx_chol_solver::ApproxCholSolver;
use super::block_elim_solver::BlockElimSolver;

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
