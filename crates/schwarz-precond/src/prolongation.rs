//! The prolongation `Rᵀ = [R₁ᵀ … Rₖᵀ]` of the stacked restriction, summing in a fixed order.

use rayon::prelude::*;

/// For each DOF, the slots of the stacked local vector holding it, in ascending slot order.
pub(crate) struct Prolongation {
    offsets: Vec<usize>,
    slots: Vec<u32>,
}

impl Prolongation {
    /// `dofs` yields each slot's DOF in slot order; it is walked twice, so it must clone cheaply.
    pub(crate) fn new(dofs: impl Iterator<Item = u32> + Clone, n_dofs: usize) -> Self {
        let mut offsets = vec![0usize; n_dofs + 1];
        for g in dofs.clone() {
            offsets[g as usize + 1] += 1;
        }
        for g in 0..n_dofs {
            offsets[g + 1] += offsets[g];
        }
        let mut next = offsets[..n_dofs].to_vec();
        let mut slots = vec![0u32; offsets[n_dofs]];
        for (s, g) in dofs.enumerate() {
            let at = &mut next[g as usize];
            slots[*at] = u32::try_from(s).expect("at most u32::MAX slots");
            *at += 1;
        }
        Self { offsets, slots }
    }

    pub(crate) fn n_dofs(&self) -> usize {
        self.offsets.len() - 1
    }

    /// `global[g] = Σ local[s]` over the slots `s` of `g`, summed in ascending `s`.
    pub(crate) fn apply(&self, local: &[f64], global: &mut [f64]) {
        debug_assert_eq!(local.len(), self.slots.len());
        debug_assert_eq!(global.len() + 1, self.offsets.len());
        const CHUNK: usize = 4096;
        global
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let first = ci * CHUNK;
                for (g, o) in (first..).zip(chunk.iter_mut()) {
                    let mut sum = 0.0;
                    for &s in &self.slots[self.offsets[g]..self.offsets[g + 1]] {
                        sum += local[s as usize];
                    }
                    *o = sum;
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::assert_pool_independent;
    use rstest::{fixture, rstest};

    #[test]
    fn sums_each_dof_in_slot_order() {
        let big = 2f64.powi(53);
        let p = Prolongation::new([1u32, 0, 1, 1].into_iter(), 3);
        let mut out = vec![f64::NAN; 3];
        p.apply(&[1.0, 3.0, big, -big], &mut out);
        // `1 + 2⁵³` rounds to `2⁵³`; summed in reverse, DOF 1 would be 1.
        assert_eq!(out, vec![3.0, 0.0, 0.0]);
    }

    const SLOTS: u64 = 400_000;
    const DOFS: u64 = 50_000;

    /// Eight slots per DOF on average, hashed so each DOF's slots lie far apart.
    #[fixture]
    fn scattered_prolongation() -> Prolongation {
        let dofs = (0..SLOTS).map(|s| ((s * 2_654_435_761) % DOFS) as u32);
        Prolongation::new(dofs, DOFS as usize)
    }

    /// Entries spanning 30 decades, so a reordered sum rounds differently.
    #[fixture]
    fn wide_range_local() -> Vec<f64> {
        (0..SLOTS as usize)
            .map(|i| ((i * 7919) % 997) as f64 * 10f64.powi((i % 31) as i32 - 15))
            .collect()
    }

    #[rstest]
    fn the_result_does_not_depend_on_the_pool(
        scattered_prolongation: Prolongation,
        wide_range_local: Vec<f64>,
    ) {
        assert_pool_independent(|| {
            let mut global = vec![0.0; DOFS as usize];
            scattered_prolongation.apply(&wide_range_local, &mut global);
            global.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        });
    }
}
