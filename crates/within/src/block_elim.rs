//! Bipartite block-elimination local solver for Schwarz factor-pair subdomains.
//!
//! [`BlockElimSolver`] handles a 2x2 bipartite Gramian `[D_q, C; C^T, D_r]` by
//! eliminating the larger diagonal block, then solving the Schur complement
//! via approximate or dense Cholesky (see [`schur`] and [`factor`]).

pub(crate) mod csr_matrix;
pub(crate) mod elimination;
pub(crate) mod factor;
pub(crate) mod schur;
pub(crate) mod solver;

pub(crate) use solver::BlockElimSolver;

/// Neumaier compensated sum: a flat `iter().sum()` loses low-order bits for large `n`, biasing the mean or ground charge derived from it.
#[inline]
pub(crate) fn compensated_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for &value in values {
        let next = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - next) + value;
        } else {
            compensation += (value - next) + sum;
        }
        sum = next;
    }
    sum + compensation
}

#[cfg(test)]
mod tests;
