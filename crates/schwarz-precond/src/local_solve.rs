//! Local solver trait and subdomain entry.
//!
//! In the Schwarz formula `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`, this module
//! provides the two key abstractions:
//!
//! - [`LocalSolver`] — the `Aᵢ⁻¹` operator: given a local right-hand
//!   side, produce an approximate (or exact) local solution. Implementations
//!   are problem-specific (e.g. approximate Cholesky, block elimination).
//!
//! - [`SubdomainEntry`] — bundles a [`SubdomainCore`] (which implements
//!   `Rᵢ` and `D̃ᵢ`) with a `LocalSolver` (which implements `Aᵢ⁻¹`),
//!   giving a single object that can compute the full per-subdomain
//!   contribution `Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ r` via [`SubdomainEntry::apply_weighted_into_with_scratch`].

use std::sync::atomic::AtomicU64;

use crate::domain::SubdomainCore;
use crate::error::{BuildError, SolveError};

// ---------------------------------------------------------------------------
// LocalSolver trait
// ---------------------------------------------------------------------------

/// The `Aᵢ⁻¹` operator in the Schwarz formula: a local subdomain solver.
///
/// Each subdomain's restricted system `Aᵢ = Rᵢ A Rᵢᵀ` is solved (exactly or
/// approximately) by an implementor of this trait. Given a right-hand side
/// in `rhs`, it writes the solution into `sol`. Both buffers are scratch-sized
/// (may be larger than `n_local` for augmented systems).
pub trait LocalSolver: Send + Sync {
    /// Number of DOFs in the subdomain (before augmentation).
    fn n_local(&self) -> usize;

    /// Required scratch buffer size (may be > n_local for augmented systems).
    fn scratch_size(&self) -> usize;

    /// Solve the local system: rhs → sol.
    ///
    /// Both `rhs` and `sol` have length >= `scratch_size()`.
    /// The solver may read/write up to `scratch_size()` elements.
    ///
    /// `allow_inner_parallelism` is a hint from the Schwarz scheduler: when
    /// `false`, the caller does not want this solve to spawn nested Rayon work.
    /// Solvers without nested-parallel regions can ignore the parameter.
    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), SolveError>;

    /// Estimated amount of local work that can benefit from nested Rayon.
    ///
    /// Return zero when the solver has no nested-parallel region worth enabling.
    fn inner_parallelism_work_estimate(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// SubdomainEntry<S>
// ---------------------------------------------------------------------------

/// One term of the Schwarz sum: `Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`.
///
/// Bundles a [`SubdomainCore`] (which provides `Rᵢ` and `D̃ᵢ`) with a
/// [`LocalSolver`] (which provides `Aᵢ⁻¹`). The
/// [`apply_weighted_into_with_scratch`](Self::apply_weighted_into_with_scratch)
/// method computes the full contribution for this subdomain.
#[derive(Clone)]
pub struct SubdomainEntry<S: LocalSolver> {
    /// Subdomain core with restriction indices and partition-of-unity weights.
    core: SubdomainCore,
    /// The local solver for this subdomain.
    solver: S,
}

impl<S: LocalSolver> SubdomainEntry<S> {
    /// Create a validated subdomain entry from a core and a local solver.
    pub fn try_new(core: SubdomainCore, solver: S) -> Result<Self, BuildError> {
        let index_count = core.n_local();
        let solver_n_local = solver.n_local();
        if solver_n_local != index_count {
            return Err(BuildError::LocalDofCountMismatch {
                index_count,
                solver_n_local,
            });
        }

        let scratch_size = solver.scratch_size();
        if scratch_size < index_count {
            return Err(BuildError::ScratchSizeTooSmall {
                scratch_size,
                required_min: index_count,
            });
        }

        Ok(Self { core, solver })
    }

    /// Subdomain metadata and partition weights.
    pub fn core(&self) -> &SubdomainCore {
        &self.core
    }

    /// Global DOF indices covered by this subdomain.
    pub fn global_indices(&self) -> &[u32] {
        self.core.global_indices()
    }

    /// Partition-of-unity weights for this subdomain.
    pub fn partition_weights(&self) -> &crate::domain::PartitionWeights {
        self.core.partition_weights()
    }

    /// The local solver for this subdomain.
    pub fn solver(&self) -> &S {
        &self.solver
    }

    /// Returns `true` if this subdomain is empty.
    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    /// Required scratch buffer size (delegates to solver).
    #[inline]
    pub fn scratch_size(&self) -> usize {
        self.solver.scratch_size()
    }

    /// Accumulate the two-sided PoU weighted local solve into a global buffer.
    ///
    /// out += R_i^T D_i A_i^{-1} D_i R_i r
    ///
    /// - `r_scratch` must have length >= `self.scratch_size()`
    /// - `z_scratch` must have length >= `self.scratch_size()`
    pub fn apply_weighted_into_with_scratch(
        &self,
        r: &[f64],
        out: &mut [f64],
        r_scratch: &mut [f64],
        z_scratch: &mut [f64],
    ) -> Result<(), SolveError> {
        self.apply_weighted_into_with_scratch_with(r, out, r_scratch, z_scratch, true)
    }

    pub(crate) fn apply_weighted_into_with_scratch_with(
        &self,
        r: &[f64],
        out: &mut [f64],
        r_scratch: &mut [f64],
        z_scratch: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), SolveError> {
        if self.core.is_empty() {
            return Ok(());
        }

        // Gather with partition weights: r_scratch = D_i @ R_i @ r
        self.core.restrict_weighted(r, r_scratch);

        // Local solve (strategy-specific transforms happen inside the solver)
        self.solver
            .solve_local(r_scratch, z_scratch, allow_inner_parallelism)?;

        // Weighted scatter directly into output: out += R_i^T @ D_i @ z_local
        self.core.prolongate_weighted_add(z_scratch, out);
        Ok(())
    }

    /// Accumulate the two-sided PoU weighted local solve into an atomic global buffer.
    ///
    /// out\[i\] += R_i^T D_i A_i^{-1} D_i R_i r  (via atomic f64 add)
    pub fn apply_weighted_into_atomic(
        &self,
        r: &[f64],
        out: &[AtomicU64],
        r_scratch: &mut [f64],
        z_scratch: &mut [f64],
    ) -> Result<(), SolveError> {
        self.apply_weighted_into_atomic_with(r, out, r_scratch, z_scratch, true)
    }

    pub(crate) fn apply_weighted_into_atomic_with(
        &self,
        r: &[f64],
        out: &[AtomicU64],
        r_scratch: &mut [f64],
        z_scratch: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), SolveError> {
        if self.core.is_empty() {
            return Ok(());
        }

        // Gather with partition weights: r_scratch = D_i @ R_i @ r
        self.core.restrict_weighted(r, r_scratch);

        // Local solve (strategy-specific transforms happen inside the solver)
        self.solver
            .solve_local(r_scratch, z_scratch, allow_inner_parallelism)?;

        // Weighted atomic scatter into output: out += R_i^T @ D_i @ z_local
        self.core.prolongate_weighted_add_atomic(z_scratch, out);
        Ok(())
    }
}

#[cfg(feature = "serde")]
impl<S> serde::Serialize for SubdomainEntry<S>
where
    S: LocalSolver + serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("SubdomainEntry", 2)?;
        state.serialize_field("core", &self.core)?;
        state.serialize_field("solver", &self.solver)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, S> serde::Deserialize<'de> for SubdomainEntry<S>
where
    S: LocalSolver + serde::de::DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(bound(deserialize = "S: serde::de::DeserializeOwned"))]
        struct Helper<S> {
            core: SubdomainCore,
            solver: S,
        }

        let helper = Helper::<S>::deserialize(deserializer)?;
        Self::try_new(helper.core, helper.solver).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PartitionWeights;

    /// Identity local solver: sol = rhs.
    struct IdentityLocalSolver {
        n: usize,
    }

    impl LocalSolver for IdentityLocalSolver {
        fn n_local(&self) -> usize {
            self.n
        }
        fn scratch_size(&self) -> usize {
            self.n
        }
        fn solve_local(
            &self,
            rhs: &mut [f64],
            sol: &mut [f64],
            _allow_inner_parallelism: bool,
        ) -> Result<(), SolveError> {
            sol[..self.n].copy_from_slice(&rhs[..self.n]);
            Ok(())
        }
    }

    #[test]
    fn test_gather_scatter_roundtrip_uniform() {
        // Subdomain covers DOFs [1, 3, 4] out of 6 global DOFs.
        let core = SubdomainCore::uniform(vec![1, 3, 4]);
        let global = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut local = vec![0.0; 3];

        // Restrict (gather): local = global[indices]
        core.restrict_weighted(&global, &mut local);
        assert_eq!(local, vec![20.0, 40.0, 50.0]);

        // Prolongate (scatter): out[indices] += local
        let mut out = vec![0.0; 6];
        core.prolongate_weighted_add(&local, &mut out);
        assert_eq!(out, vec![0.0, 20.0, 0.0, 40.0, 50.0, 0.0]);
    }

    #[test]
    fn test_weighted_gather_scatter() {
        // Non-uniform weights: two-sided PoU means effective weight = w^2
        let core = SubdomainCore::with_partition_weights(
            vec![0, 1, 2],
            PartitionWeights::NonUniform(vec![1.0, 0.5, 0.25]),
        )
        .expect("matching non-uniform weights");
        let global = vec![4.0, 8.0, 16.0];
        let mut local = vec![0.0; 3];

        // Restrict: local[i] = w[i] * global[idx[i]]
        core.restrict_weighted(&global, &mut local);
        assert!((local[0] - 4.0).abs() < 1e-14); // 1.0 * 4.0
        assert!((local[1] - 4.0).abs() < 1e-14); // 0.5 * 8.0
        assert!((local[2] - 4.0).abs() < 1e-14); // 0.25 * 16.0

        // Prolongate: out[idx[i]] += w[i] * local[i]
        let mut out = vec![0.0; 3];
        core.prolongate_weighted_add(&local, &mut out);
        // out[0] = 1.0 * 4.0 = 4.0,  out[1] = 0.5 * 4.0 = 2.0,  out[2] = 0.25 * 4.0 = 1.0
        // Full round-trip effective weight = w^2: 1.0, 0.25, 0.0625
        assert!((out[0] - 4.0).abs() < 1e-14);
        assert!((out[1] - 2.0).abs() < 1e-14);
        assert!((out[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_apply_weighted_with_identity_solver() {
        // With identity solver and uniform weights:
        // apply_weighted = R_i^T D_i (I) D_i R_i r = R_i^T R_i r (since D_i = I)
        let core = SubdomainCore::uniform(vec![1, 2]);
        let solver = IdentityLocalSolver { n: 2 };
        let entry = SubdomainEntry::try_new(core, solver).expect("valid entry");

        let r = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        let mut r_scratch = vec![0.0; 2];
        let mut z_scratch = vec![0.0; 2];

        entry
            .apply_weighted_into_with_scratch(&r, &mut out, &mut r_scratch, &mut z_scratch)
            .expect("identity local solver should not fail");

        // Should restrict DOFs [1,2] → [20, 30], identity solve, scatter back
        assert_eq!(out, vec![0.0, 20.0, 30.0, 0.0]);
    }
}
