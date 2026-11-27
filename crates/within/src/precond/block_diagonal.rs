//! PROTOTYPE(#607): block Jacobi on `S`'s level blocks: one `(1 + slopes)²` block per kept term
//! and level, pseudo-inverted. On an intercept-only term each block is `diag(S)`'s entry, inverted
//! exactly as the diagonal rung does.

use std::sync::Arc;

use faer::{Mat, Side};
use rayon::prelude::*;
use schwarz_precond::{MemStack, Operator, SolveError};
use serde::{Deserialize, Serialize};

use crate::system::{Gramian, TermLevelBlocks};
use crate::BuildError;

/// A scaled block's eigenvalue at or below this share of its largest is null: dropped by the
/// pseudo-inverse. The scaling gives each block a unit diagonal, so the cutoff is on the block's
/// conditioning, not its scale.
const EIGEN_CUTOFF: f64 = 1e-12;

/// Levels per parallel task of an apply.
const LEVEL_CHUNK: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct BlockDiagonalPreconditioner {
    n: usize,
    terms: Arc<[TermInverse]>,
}

/// One kept term's inverted level blocks, laid out like [`TermLevelBlocks`].
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TermInverse {
    start: usize,
    n_levels: usize,
    width: usize,
    inverse: Vec<f64>,
}

pub(super) fn build(gramian: Gramian<'_>) -> Result<BlockDiagonalPreconditioner, BuildError> {
    let terms = gramian
        .level_blocks()
        .into_iter()
        .map(invert_term)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BlockDiagonalPreconditioner {
        n: gramian.n(),
        terms: terms.into(),
    })
}

fn invert_term(term: TermLevelBlocks) -> Result<TermInverse, BuildError> {
    let TermLevelBlocks {
        start,
        n_levels,
        width: p,
        mut blocks,
    } = term;
    if p == 1 {
        // As the diagonal rung: a zero diagonal is an unidentified unknown, kept at 0.
        for (level, d) in blocks.iter_mut().enumerate() {
            if *d == 0.0 {
                continue;
            }
            let inv = 1.0 / *d;
            if !inv.is_finite() {
                return Err(BuildError::SingularDiagonal {
                    index: start + level,
                });
            }
            *d = inv;
        }
    } else {
        blocks
            .par_chunks_exact_mut(p * p)
            .with_min_len(256)
            .for_each(|block| pseudo_invert(block, p));
    }
    Ok(TermInverse {
        start,
        n_levels,
        width: p,
        inverse: blocks,
    })
}

/// `block ← D (D block D)⁺ D` with `D` scaling the block to a unit diagonal; a zero diagonal's row
/// and column stay zero.
fn pseudo_invert(block: &mut [f64], p: usize) {
    let scale: Vec<f64> = (0..p)
        .map(|q| {
            let d = block[q * p + q];
            if d > 0.0 {
                d.sqrt().recip()
            } else {
                0.0
            }
        })
        .collect();
    let scaled = Mat::from_fn(p, p, |a, b| block[a * p + b] * scale[a] * scale[b]);
    block.fill(0.0);
    let Ok(eigen) = scaled.self_adjoint_eigen(Side::Lower) else {
        return;
    };
    let (values, vectors) = (eigen.S(), eigen.U());
    let max = (0..p).map(|i| values[i]).fold(0.0f64, f64::max);
    for i in 0..p {
        let lambda = values[i];
        if lambda <= EIGEN_CUTOFF * max {
            continue;
        }
        for a in 0..p {
            for b in 0..p {
                block[a * p + b] += vectors[(a, i)] * vectors[(b, i)] / lambda;
            }
        }
    }
    for a in 0..p {
        for b in 0..p {
            block[a * p + b] *= scale[a] * scale[b];
        }
    }
}

impl Operator for BlockDiagonalPreconditioner {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn apply_with(&self, x: &[f64], y: &mut [f64], _: &mut MemStack) -> Result<(), SolveError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(SolveError::InvalidInput {
                context: "BlockDiagonalPreconditioner::apply",
                message: format!(
                    "x.len()={}, y.len()={}, expected n_dofs={}",
                    x.len(),
                    y.len(),
                    self.n
                ),
            });
        }
        y.fill(0.0);
        let mut rest = &mut y[..];
        let mut at = 0;
        for term in self.terms.iter() {
            let (l, p) = (term.n_levels, term.width);
            let (_, tail) = rest.split_at_mut(term.start - at);
            let (block, tail) = tail.split_at_mut(l * p);
            (rest, at) = (tail, term.start + l * p);
            let xs = &x[term.start..][..l * p];
            if p == 1 {
                (block.par_iter_mut().zip(xs).zip(&term.inverse))
                    .with_min_len(LEVEL_CHUNK)
                    .for_each(|((y, &x), &d)| *y = d * x);
                continue;
            }
            // Column `q` of the term's block is `block[q·l..(q+1)·l]`: cut every column at the
            // same level chunks, so each task owns its levels' rows of every column.
            let mut columns: Vec<_> = block
                .chunks_mut(l)
                .map(|c| c.chunks_mut(LEVEL_CHUNK))
                .collect();
            let tasks: Vec<Vec<&mut [f64]>> = (0..l.div_ceil(LEVEL_CHUNK))
                .map(|_| {
                    (columns.iter_mut())
                        .map(|c| c.next().expect("a chunk per column"))
                        .collect()
                })
                .collect();
            tasks.into_par_iter().enumerate().for_each(|(t, mut cols)| {
                let first = t * LEVEL_CHUNK;
                let mut xl = vec![0.0; p];
                for k in 0..cols[0].len() {
                    let level = first + k;
                    for (q, v) in xl.iter_mut().enumerate() {
                        *v = xs[q * l + level];
                    }
                    let inv = &term.inverse[level * p * p..][..p * p];
                    for (q, col) in cols.iter_mut().enumerate() {
                        col[k] = (0..p).map(|r| inv[q * p + r] * xl[r]).sum();
                    }
                }
            });
        }
        Ok(())
    }

    fn apply_adjoint_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.apply_with(x, y, stack)
    }
}
