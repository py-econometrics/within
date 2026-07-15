//! Gather kernel: coefficient space → observation space (`D x`).

use rayon::prelude::*;

use super::PAR_THRESHOLD;
use crate::domain::Design;

/// Gather-apply: `dst[i] = Σ_t Σ_c src[off_t + c·L_t + level(i,t)] · loading_c(i)`,
/// times `scale[i]` if given (loading is `1` for intercept columns).
pub(crate) fn gather_apply(
    design: &Design<'_>,
    src: &[f64],
    dst: &mut [f64],
    scale: Option<&[f64]>,
) {
    debug_assert!(scale.is_none_or(|s| s.len() == design.n_obs));
    debug_assert_eq!(src.len(), design.n_dofs);
    debug_assert_eq!(dst.len(), design.n_obs);

    dst.fill(0.0);

    let frame = &design.frame;
    for_each_chunk(dst, |chunk, row_start| {
        for (q, t) in design.terms.iter().enumerate() {
            let (offset, n_levels) = (t.offset, t.n_levels);
            let levels = frame.level_column(q);
            let col = |c: usize| &src[offset + c * n_levels..offset + (c + 1) * n_levels];
            match (t.intercept, t.slopes.as_slice()) {
                (true, []) => gather_term(chunk, row_start, levels, [col(0)], |_| [1.0]),
                (true, &[c0]) => {
                    let z0 = frame.loading_column(c0);
                    gather_term(chunk, row_start, levels, [col(0), col(1)], |i| [1.0, z0[i]])
                }
                (true, &[c0, c1]) => {
                    let z0 = frame.loading_column(c0);
                    let z1 = frame.loading_column(c1);
                    gather_term(chunk, row_start, levels, [col(0), col(1), col(2)], |i| {
                        [1.0, z0[i], z1[i]]
                    })
                }
                (false, &[c0]) => {
                    let z0 = frame.loading_column(c0);
                    gather_term(chunk, row_start, levels, [col(0)], |i| [z0[i]])
                }
                (intercept, slope_cols) => {
                    // Cold path: a dynamic slope count can't monomorphize a
                    // fixed-arity `gather_term`, so accumulate by hand.
                    let zoff = usize::from(intercept);
                    for (local, dst_val) in chunk.iter_mut().enumerate() {
                        let i = row_start + local;
                        let lev = levels[i] as usize;
                        let mut acc = if intercept { src[offset + lev] } else { 0.0 };
                        for (v, &c) in slope_cols.iter().enumerate() {
                            acc += src[offset + (zoff + v) * n_levels + lev]
                                * frame.loading_column(c)[i];
                        }
                        *dst_val += acc;
                    }
                }
            }
        }
        if let Some(scale) = scale {
            for (s, dst_val) in scale[row_start..].iter().zip(chunk.iter_mut()) {
                *dst_val *= s;
            }
        }
    });
}

/// Sweep `dst` in cache-sized chunks, in parallel above [`PAR_THRESHOLD`] rows;
/// the kernel receives each chunk and the row index it starts at.
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

/// One term sweep with a compile-time column count: `chunk[local] += Σ_c
/// cols[c][level(i)] · weights(i)[c]`.
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
