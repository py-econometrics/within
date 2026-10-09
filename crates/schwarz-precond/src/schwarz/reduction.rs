//! Ordered incidence of subdomain contributions at each global degree of freedom.

use rayon::prelude::*;

use crate::{LocalSolver, SubdomainEntry};

pub(super) struct ReductionLayout {
    offsets: Vec<usize>,
    sources: Vec<(usize, usize)>,
}

impl ReductionLayout {
    pub(super) fn new<S: LocalSolver>(entries: &[SubdomainEntry<S>], n_dofs: usize) -> Self {
        let mut offsets = vec![0; n_dofs + 1];
        for entry in entries {
            for &dof in entry.global_indices() {
                offsets[dof as usize + 1] += 1;
            }
        }
        for dof in 0..n_dofs {
            offsets[dof + 1] += offsets[dof];
        }
        let mut sources = vec![(0, 0); offsets[n_dofs]];
        let mut cursors = offsets[..n_dofs].to_vec();
        for (subdomain, entry) in entries.iter().enumerate() {
            for (local, &dof) in entry.global_indices().iter().enumerate() {
                let cursor = &mut cursors[dof as usize];
                sources[*cursor] = (subdomain, local);
                *cursor += 1;
            }
        }
        Self { offsets, sources }
    }

    pub(super) fn sum(&self, outputs: &[Vec<f64>], z: &mut [f64]) {
        const CHUNK: usize = 4096;
        z.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(block, chunk)| {
                for (within, value) in chunk.iter_mut().enumerate() {
                    let dof = block * CHUNK + within;
                    *value = self.sources[self.offsets[dof]..self.offsets[dof + 1]]
                        .iter()
                        .map(|&(subdomain, local)| outputs[subdomain][local])
                        .fold(0.0, |sum, value| sum + value);
                }
            });
    }
}
