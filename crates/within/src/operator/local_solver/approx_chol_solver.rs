use approx_chol::Factor;
use schwarz_precond::{LocalSolveError, LocalSolver};

use super::transforms::{negate_block, subtract_mean};
use super::LocalSolveStrategy;

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
