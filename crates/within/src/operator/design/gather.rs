//! Gather kernel: coefficient space → observation space (`D x`).

use rayon::prelude::*;

use super::{resolve_terms, PAR_THRESHOLD};
use crate::domain::Design;

/// Gather-apply: `dst[i] = Σ_t Σ_c src[off_t + c·L_t + level(i,t)] · loading_c(i)`,
/// times `scale[i]` if given (loading is `1` for intercept columns).
///
/// One sweep over `dst` per term — a term's columns share one level load and
/// their gathers overlap in the load queue, measured faster than per-column
/// sweeps on unsorted factors and a tie elsewhere — plus a scale sweep only
/// when given.
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

    let terms = resolve_terms(design);

    let kernel = |chunk: &mut [f64], row_start: usize| {
        for t in &terms {
            let o = t.meta.offset;
            let l = t.meta.n_levels;
            // Copy the slice ref out of the descriptor; a double deref in the
            // inner loop cost ~5% historically.
            let levels = t.levels;
            match (t.meta.intercept, t.zs.as_slice()) {
                (true, []) => {
                    for (local, dst_val) in chunk.iter_mut().enumerate() {
                        let i = row_start + local;
                        *dst_val += src[o + levels[i] as usize];
                    }
                }
                (true, &[z0]) => {
                    for (local, dst_val) in chunk.iter_mut().enumerate() {
                        let i = row_start + local;
                        let lev = levels[i] as usize;
                        *dst_val += src[o + lev] + src[o + l + lev] * z0[i];
                    }
                }
                (true, &[z0, z1]) => {
                    for (local, dst_val) in chunk.iter_mut().enumerate() {
                        let i = row_start + local;
                        let lev = levels[i] as usize;
                        *dst_val +=
                            src[o + lev] + src[o + l + lev] * z0[i] + src[o + 2 * l + lev] * z1[i];
                    }
                }
                (false, &[z0]) => {
                    for (local, dst_val) in chunk.iter_mut().enumerate() {
                        let i = row_start + local;
                        *dst_val += src[o + levels[i] as usize] * z0[i];
                    }
                }
                (intercept, zs) => {
                    let zoff = usize::from(intercept);
                    for (local, dst_val) in chunk.iter_mut().enumerate() {
                        let i = row_start + local;
                        let lev = levels[i] as usize;
                        let mut acc = if intercept { src[o + lev] } else { 0.0 };
                        for (v, z) in zs.iter().enumerate() {
                            acc += src[o + (zoff + v) * l + lev] * z[i];
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
    };

    if design.n_obs > PAR_THRESHOLD {
        const CHUNK_SIZE: usize = 4096;

        dst.par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                kernel(chunk, chunk_idx * CHUNK_SIZE);
            });
    } else {
        kernel(dst, 0);
    }
}
