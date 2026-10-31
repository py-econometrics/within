//! Domain-decomposition primitives.
//!
//! - [`SubdomainCore`] — restriction `Rᵢ` and partition-of-unity weighting `D̃ᵢ`
//!   for one subdomain; paired with a [`LocalSolver`](crate::local_solve::LocalSolver)
//!   to form a Schwarz term `Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`.
//! - [`PartitionWeights`] — per-DOF weights, uniform constant or explicit vector.

use crate::error::BuildError;

/// PoU weights for a subdomain; the common uniform-1.0 case avoids allocating.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub enum PartitionWeights {
    /// All weights are 1.0 (no shared DOFs).
    Uniform,
    /// Non-uniform weights (1/√count for shared DOFs).
    NonUniform(Vec<f64>),
}

impl PartitionWeights {
    /// Get the weight at index `i`.
    pub fn get(&self, i: usize) -> f64 {
        match self {
            Self::Uniform => 1.0,
            Self::NonUniform(w) => w[i],
        }
    }
}

/// Subdomain `Rᵢ` and `D̃ᵢ`: sorted global DOF indices plus partition-of-unity weights.
#[derive(Clone)]
pub struct SubdomainCore {
    /// Global DOF indices belonging to this subdomain.
    global_indices: Vec<u32>,
    /// Partition-of-unity weights for each DOF.
    partition_weights: PartitionWeights,
}

impl SubdomainCore {
    /// Create a subdomain core with uniform weights.
    pub fn uniform(global_indices: Vec<u32>) -> Self {
        Self {
            partition_weights: PartitionWeights::Uniform,
            global_indices,
        }
    }

    /// Create a subdomain core with explicit partition weights.
    pub fn with_partition_weights(
        global_indices: Vec<u32>,
        partition_weights: PartitionWeights,
    ) -> Result<Self, BuildError> {
        let mut core = Self::uniform(global_indices);
        core.set_partition_weights(partition_weights)?;
        Ok(core)
    }

    /// Global DOF indices belonging to this subdomain.
    pub fn global_indices(&self) -> &[u32] {
        &self.global_indices
    }

    /// Partition-of-unity weights for this subdomain.
    pub fn partition_weights(&self) -> &PartitionWeights {
        &self.partition_weights
    }

    /// Number of local DOFs in this subdomain.
    pub fn n_local(&self) -> usize {
        self.global_indices.len()
    }

    /// Returns `true` if this subdomain has no DOFs.
    pub fn is_empty(&self) -> bool {
        self.global_indices.is_empty()
    }

    /// Replace the partition weights, validating that the shape matches.
    pub fn set_partition_weights(
        &mut self,
        partition_weights: PartitionWeights,
    ) -> Result<(), BuildError> {
        if let PartitionWeights::NonUniform(weights) = &partition_weights {
            let index_count = self.global_indices.len();
            let weight_count = weights.len();
            if weight_count != index_count {
                return Err(BuildError::PartitionWeightLengthMismatch {
                    index_count,
                    weight_count,
                });
            }
        }
        self.partition_weights = partition_weights;
        Ok(())
    }

    /// Reset the partition weights to the uniform representation.
    pub fn set_uniform_partition_weights(&mut self) {
        self.partition_weights = PartitionWeights::Uniform;
    }

    /// Weighted gather: `local[i] = w[i] * global[idx[i]]`
    #[inline]
    pub(crate) fn restrict_weighted(&self, global: &[f64], local: &mut [f64]) {
        match &self.partition_weights {
            PartitionWeights::Uniform => {
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

    /// Weighting in local coordinates: `local[i] *= w[i]`
    #[inline]
    pub(crate) fn weight(&self, local: &mut [f64]) {
        if let PartitionWeights::NonUniform(w) = &self.partition_weights {
            for (li, &wi) in local.iter_mut().zip(w) {
                *li *= wi;
            }
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SubdomainCore {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("SubdomainCore", 2)?;
        state.serialize_field("global_indices", &self.global_indices)?;
        state.serialize_field("partition_weights", &self.partition_weights)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SubdomainCore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            global_indices: Vec<u32>,
            partition_weights: PartitionWeights,
        }

        let helper = Helper::deserialize(deserializer)?;
        Self::with_partition_weights(helper.global_indices, helper.partition_weights)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_weights_take_the_shape_from_the_indices() {
        for indices in [vec![], vec![2], vec![1, 3, 7]] {
            let core =
                SubdomainCore::with_partition_weights(indices.clone(), PartitionWeights::Uniform)
                    .expect("uniform weights fit any index count");
            assert_eq!(core.global_indices(), indices);
            assert_eq!(core.n_local(), indices.len());
            assert_eq!(core.is_empty(), indices.is_empty());
            for i in 0..core.n_local() {
                assert_eq!(core.partition_weights().get(i), 1.0);
            }
        }
    }

    #[test]
    fn invalid_explicit_shape_does_not_replace_the_weights() {
        let mut core = SubdomainCore::uniform(vec![0, 2]);
        assert!(matches!(
            core.set_partition_weights(PartitionWeights::NonUniform(vec![0.5])),
            Err(BuildError::PartitionWeightLengthMismatch {
                index_count: 2,
                weight_count: 1,
            })
        ));
        assert!(matches!(
            core.partition_weights(),
            PartitionWeights::Uniform
        ));
        core.set_partition_weights(PartitionWeights::NonUniform(vec![0.5, 2.0]))
            .expect("matching explicit weights");
        core.set_uniform_partition_weights();
        assert!(matches!(
            core.partition_weights(),
            PartitionWeights::Uniform
        ));
        assert_eq!(core.partition_weights().get(1), 1.0);
    }

    #[test]
    fn uniform_and_explicit_weighted_operations_keep_their_values() {
        let mut core = SubdomainCore::uniform(vec![0, 2, 4]);
        let global = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut local = [0.0; 3];
        core.restrict_weighted(&global, &mut local);
        core.weight(&mut local);
        assert_eq!(local, [1.0, 3.0, 5.0]);
        core.set_partition_weights(PartitionWeights::NonUniform(vec![0.5, 1.0, 2.0]))
            .expect("matching explicit weights");
        core.restrict_weighted(&global, &mut local);
        core.weight(&mut local);
        assert_eq!(local, [0.25, 3.0, 20.0]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn uniform_wire_value_has_no_count() {
        assert_eq!(
            postcard::to_stdvec(&PartitionWeights::Uniform).unwrap(),
            [0]
        );
        let core = SubdomainCore::uniform(vec![1, 3]);
        let bytes = postcard::to_stdvec(&core).unwrap();
        let restored: SubdomainCore = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored.global_indices(), [1, 3]);
        assert!(matches!(
            restored.partition_weights(),
            PartitionWeights::Uniform
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialization_still_checks_explicit_weight_lengths() {
        #[derive(serde::Serialize)]
        struct InvalidCore {
            global_indices: Vec<u32>,
            partition_weights: PartitionWeights,
        }
        let matching = postcard::to_stdvec(&InvalidCore {
            global_indices: vec![0, 1],
            partition_weights: PartitionWeights::NonUniform(vec![0.5, 2.0]),
        })
        .unwrap();
        let restored: SubdomainCore = postcard::from_bytes(&matching).unwrap();
        assert_eq!(restored.global_indices(), [0, 1]);
        assert_eq!(restored.partition_weights().get(0), 0.5);
        assert_eq!(restored.partition_weights().get(1), 2.0);
        let mismatching = postcard::to_stdvec(&InvalidCore {
            global_indices: vec![0, 1],
            partition_weights: PartitionWeights::NonUniform(vec![0.5]),
        })
        .unwrap();
        assert!(postcard::from_bytes::<SubdomainCore>(&mismatching).is_err());
    }
}
