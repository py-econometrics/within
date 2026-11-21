//! Gather: coefficient space → observation space, `y = s ⊙ Σ_t D_t x_t`.

use rayon::prelude::*;

use crate::design::{RowKernel, TermBlock};
use crate::system::OrthogonalDesign;
use crate::PAR_THRESHOLD;

/// `dst = D̃ src`, row `i` times `scale[i]` when given.
pub(in crate::system) fn gather(
    design: &OrthogonalDesign<'_>,
    src: &[f64],
    dst: &mut [f64],
    scale: Option<&[f64]>,
) {
    debug_assert!(scale.is_none_or(|s| s.len() == dst.len()));
    debug_assert_eq!(design.n_obs(), dst.len());
    debug_assert_eq!(src.len(), design.unknowns().len());

    dst.fill(0.0);

    for_each_chunk(dst, |chunk, row_start| {
        for (t, term) in design.terms().enumerate() {
            term.visit_rows(GatherTerm {
                chunk: &mut *chunk,
                row_start,
                block: &src[design.unknowns().block(t)],
                term: &term,
            });
        }
        if let Some(scale) = scale {
            for (s, dst_val) in scale[row_start..].iter().zip(chunk.iter_mut()) {
                *dst_val *= s;
            }
        }
    });
}

/// Sweep `dst` in cache-sized chunks, parallel above [`PAR_THRESHOLD`] rows.
fn for_each_chunk(dst: &mut [f64], kernel: impl Fn(&mut [f64], usize) + Sync) {
    if dst.len() > PAR_THRESHOLD {
        const CHUNK_SIZE: usize = 4096;

        dst.par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| kernel(chunk, chunk_idx * CHUNK_SIZE));
    } else {
        kernel(dst, 0);
    }
}

/// One term's gather into the rows `row_start..` of `chunk`.
struct GatherTerm<'k, 't> {
    chunk: &'k mut [f64],
    row_start: usize,
    /// The term's block of `x`.
    block: &'k [f64],
    term: &'k TermBlock<'t>,
}

impl RowKernel for GatherTerm<'_, '_> {
    #[inline(always)]
    fn fixed<const C: usize>(self, row: impl Fn(usize) -> [f64; C] + Sync) {
        let (layout, block) = (self.term.layout(), self.block);
        let cols = std::array::from_fn(|c| &block[layout.block_column(c)]);
        gather_term(self.chunk, self.row_start, self.term.levels(), cols, row);
    }

    #[inline(always)]
    fn dynamic(self) {
        // A dynamic column count cannot monomorphize a fixed arity.
        let (t, levels) = (self.term, self.term.levels());
        let block = self.block;
        for (local, dst_val) in self.chunk.iter_mut().enumerate() {
            let i = self.row_start + local;
            let lev = levels[i] as usize;
            let mut acc = 0.0;
            for c in 0..t.layout().n_columns() {
                let coef = block[t.layout().block_column(c)][lev];
                acc += match t.loading(c) {
                    None => coef,
                    Some(z) => coef * z[i],
                };
            }
            *dst_val += acc;
        }
    }
}

/// One term sweep with a compile-time column count.
#[inline(always)]
fn gather_term<const N: usize>(
    chunk: &mut [f64],
    row_start: usize,
    levels: &[u32],
    cols: [&[f64]; N],
    weights: impl Fn(usize) -> [f64; N],
) {
    for (local, dst_val) in chunk.iter_mut().enumerate() {
        let i = row_start + local;
        let lev = levels[i] as usize;
        let row = cols.iter().zip(weights(i)).map(|(col, w)| col[lev] * w);
        // Fold from -0.0, not 0.0: the true additive identity, folds away.
        *dst_val += row.fold(-0.0, |acc, term| acc + term);
    }
}
