use std::sync::atomic::{AtomicU64, Ordering};

/// Partition-of-unity weights for a subdomain.
///
/// Most subdomains have uniform weights (all 1.0), so we avoid allocating
/// a full `Vec<f64>` for them.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub enum PartitionWeights {
    /// All weights are 1.0 (no shared DOFs).
    Uniform(usize),
    /// Non-uniform weights (1/count for shared DOFs).
    NonUniform(Vec<f64>),
}

impl PartitionWeights {
    /// Get the weight at index `i`.
    pub fn get(&self, i: usize) -> f64 {
        match self {
            Self::Uniform(_) => 1.0,
            Self::NonUniform(w) => w[i],
        }
    }

    /// Number of weights (= number of DOFs in the subdomain).
    pub fn len(&self) -> usize {
        match self {
            Self::Uniform(n) => *n,
            Self::NonUniform(w) => w.len(),
        }
    }

    /// Returns `true` if there are no weights.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Atomic add for f64 using CAS loop on `AtomicU64` bit representation.
#[inline(always)]
fn atomic_f64_add(target: &AtomicU64, val: f64) {
    let mut old_bits = target.load(Ordering::Relaxed);
    loop {
        let new_val = f64::from_bits(old_bits) + val;
        match target.compare_exchange_weak(
            old_bits,
            new_val.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => old_bits = x,
        }
    }
}

/// A domain-agnostic subdomain core: DOF indices + partition-of-unity weights.
///
/// Downstream crates (e.g. `within`) may wrap this with application-specific
/// metadata such as factor-pair information.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub struct SubdomainCore {
    /// Global DOF indices belonging to this subdomain.
    pub global_indices: Vec<u32>,
    /// Partition-of-unity weights for each DOF.
    pub partition_weights: PartitionWeights,
}

impl SubdomainCore {
    /// Create a subdomain core with uniform weights.
    pub fn uniform(global_indices: Vec<u32>) -> Self {
        let n = global_indices.len();
        SubdomainCore {
            global_indices,
            partition_weights: PartitionWeights::Uniform(n),
        }
    }

    /// Weighted gather: `local[i] = w[i] * global[idx[i]]`
    #[inline]
    pub fn restrict_weighted(&self, global: &[f64], local: &mut [f64]) {
        match &self.partition_weights {
            PartitionWeights::Uniform(_) => {
                for (dst, &gi) in local.iter_mut().zip(self.global_indices.iter()) {
                    *dst = global[gi as usize];
                }
            }
            PartitionWeights::NonUniform(w) => {
                for ((dst, &gi), &wi) in local
                    .iter_mut()
                    .zip(self.global_indices.iter())
                    .zip(w.iter())
                {
                    *dst = global[gi as usize] * wi;
                }
            }
        }
    }

    /// Weighted scatter: `global[idx[i]] += w[i] * local[i]`
    #[inline]
    pub fn prolongate_weighted_add(&self, local: &[f64], global: &mut [f64]) {
        match &self.partition_weights {
            PartitionWeights::Uniform(_) => {
                for (&gi, &li) in self.global_indices.iter().zip(local.iter()) {
                    global[gi as usize] += li;
                }
            }
            PartitionWeights::NonUniform(w) => {
                for ((&gi, &li), &wi) in self.global_indices.iter().zip(local.iter()).zip(w.iter())
                {
                    global[gi as usize] += li * wi;
                }
            }
        }
    }

    /// Weighted scatter into atomic accumulator: `global[idx[i]] += w[i] * local[i]`
    #[inline]
    pub fn prolongate_weighted_add_atomic(&self, local: &[f64], global: &[AtomicU64]) {
        match &self.partition_weights {
            PartitionWeights::Uniform(_) => {
                for (&gi, &li) in self.global_indices.iter().zip(local.iter()) {
                    atomic_f64_add(&global[gi as usize], li);
                }
            }
            PartitionWeights::NonUniform(w) => {
                for ((&gi, &li), &wi) in self.global_indices.iter().zip(local.iter()).zip(w.iter())
                {
                    atomic_f64_add(&global[gi as usize], li * wi);
                }
            }
        }
    }
}
