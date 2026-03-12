//! Local solver trait and subdomain entry.
//!
//! `LocalSolver` defines the generic A_j^{-1} abstraction.
//! `SubdomainEntry<S>` owns indices, weights, and a solver — provides gather/scatter/PoU.

use std::sync::atomic::AtomicU64;

use crate::domain::SubdomainCore;
use crate::error::LocalSolveError;

// ---------------------------------------------------------------------------
// LocalSolveOptions
// ---------------------------------------------------------------------------

/// Runtime options for one local solve invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalSolveOptions {
    allow_inner_parallelism: bool,
}

impl LocalSolveOptions {
    /// Create a new local-solve options value.
    pub const fn new(allow_inner_parallelism: bool) -> Self {
        Self {
            allow_inner_parallelism,
        }
    }

    /// Whether nested inner parallelism is allowed for this local solve.
    pub const fn allow_inner_parallelism(self) -> bool {
        self.allow_inner_parallelism
    }
}

impl Default for LocalSolveOptions {
    fn default() -> Self {
        Self::new(true)
    }
}

// ---------------------------------------------------------------------------
// LocalSolver trait
// ---------------------------------------------------------------------------

/// Trait for a local subdomain solver (the A_j^{-1} abstraction).
///
/// Implementors know how to solve the local system: given a right-hand side
/// in `rhs`, write the solution into `sol`. Both buffers are scratch-sized
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
    fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError>;

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

/// A subdomain entry: wraps a `SubdomainCore` (restriction indices + partition-of-unity
/// weights) together with a generic local solver. Delegates gather/scatter/PoU
/// weighting to `SubdomainCore`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "S: serde::Serialize",
        deserialize = "S: serde::de::DeserializeOwned"
    ))
)]
pub struct SubdomainEntry<S: LocalSolver> {
    /// Subdomain core with restriction indices and partition-of-unity weights.
    pub core: SubdomainCore,
    /// The local solver for this subdomain.
    pub solver: S,
}

impl<S: LocalSolver> SubdomainEntry<S> {
    /// Create a new subdomain entry from a core and a local solver.
    pub fn new(core: SubdomainCore, solver: S) -> Self {
        Self { core, solver }
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
        options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        if self.core.global_indices.is_empty() {
            return Ok(());
        }

        // Gather with partition weights: r_scratch = D_i @ R_i @ r
        self.core.restrict_weighted(r, r_scratch);

        // Local solve (strategy-specific transforms happen inside the solver)
        self.solver.solve_local(r_scratch, z_scratch, options)?;

        // Weighted scatter directly into output: out += R_i^T @ D_i @ z_local
        self.core.prolongate_weighted_add(z_scratch, out);
        Ok(())
    }

    /// Accumulate the two-sided PoU weighted local solve into an atomic global buffer.
    ///
    /// out[i] += R_i^T D_i A_i^{-1} D_i R_i r  (via atomic f64 add)
    pub fn apply_weighted_into_atomic(
        &self,
        r: &[f64],
        out: &[AtomicU64],
        r_scratch: &mut [f64],
        z_scratch: &mut [f64],
        options: LocalSolveOptions,
    ) -> Result<(), LocalSolveError> {
        if self.core.global_indices.is_empty() {
            return Ok(());
        }

        // Gather with partition weights: r_scratch = D_i @ R_i @ r
        self.core.restrict_weighted(r, r_scratch);

        // Local solve (strategy-specific transforms happen inside the solver)
        self.solver.solve_local(r_scratch, z_scratch, options)?;

        // Weighted atomic scatter into output: out += R_i^T @ D_i @ z_local
        self.core.prolongate_weighted_add_atomic(z_scratch, out);
        Ok(())
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
            _options: LocalSolveOptions,
        ) -> Result<(), LocalSolveError> {
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
        let core = SubdomainCore {
            global_indices: vec![0, 1, 2],
            partition_weights: PartitionWeights::NonUniform(vec![1.0, 0.5, 0.25]),
        };
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
        let entry = SubdomainEntry::new(core, solver);

        let r = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        let mut r_scratch = vec![0.0; 2];
        let mut z_scratch = vec![0.0; 2];

        entry
            .apply_weighted_into_with_scratch(
                &r,
                &mut out,
                &mut r_scratch,
                &mut z_scratch,
                LocalSolveOptions::default(),
            )
            .expect("identity local solver should not fail");

        // Should restrict DOFs [1,2] → [20, 30], identity solve, scatter back
        assert_eq!(out, vec![0.0, 20.0, 30.0, 0.0]);
    }
}
