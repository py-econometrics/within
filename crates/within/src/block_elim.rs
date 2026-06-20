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
