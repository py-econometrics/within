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
//!   giving a single object that can compute the per-subdomain
//!   contribution `D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ r` via [`SubdomainEntry::apply_weighted_local`].

use dyn_stack::{MemStack, StackReq};

use crate::domain::SubdomainCore;
use crate::error::{BuildError, LocalSolveError};
use crate::threads::Threads;

/// The `Aᵢ⁻¹` operator: solves the restricted system `Aᵢ = Rᵢ A Rᵢᵀ`, exactly or approximately.
pub trait LocalSolver: Send + Sync {
    /// Number of DOFs in the subdomain.
    fn n_local(&self) -> usize;

    /// What `solve_local` takes from its stack at `threads`.
    fn scratch(&self, threads: Threads) -> StackReq;

    /// Solves in place: `x` spans `n_local()`, rhs in, solution out; bits depend on `x` alone.
    ///
    /// `stack` holds at least `scratch(threads)`, with stale contents.
    fn solve_local(
        &self,
        x: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), LocalSolveError>;

    /// Estimated local work that can benefit from more than one thread; zero when there is none.
    fn inner_parallelism_work_estimate(&self) -> usize {
        0
    }
}

/// One term of the Schwarz sum `Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`.
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
        Ok(Self { core, solver })
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

    /// Writes `D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ r` locally to `out`; `stack` holds at least the solver's scratch.
    pub(crate) fn apply_weighted_local(
        &self,
        r: &[f64],
        out: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), LocalSolveError> {
        if self.core.is_empty() {
            return Ok(());
        }
        self.core.restrict_weighted(r, out);
        self.solver.solve_local(out, threads, stack)?;
        self.core.weight(out);
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
    use rstest::rstest;

    struct IdentityLocalSolver {
        n: usize,
    }

    impl LocalSolver for IdentityLocalSolver {
        fn n_local(&self) -> usize {
            self.n
        }
        fn scratch(&self, _threads: Threads) -> StackReq {
            StackReq::EMPTY
        }
        fn solve_local(
            &self,
            _x: &mut [f64],
            _threads: Threads,
            _stack: &mut MemStack,
        ) -> Result<(), LocalSolveError> {
            Ok(())
        }
    }

    #[rstest]
    #[case::uniform(
        SubdomainCore::uniform(vec![1, 3, 4]),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![20.0, 40.0, 50.0],
        vec![20.0, 40.0, 50.0]
    )]
    // Two-sided partition of unity: the effective weight is `w²`.
    #[case::nonuniform(
        SubdomainCore::with_partition_weights(
            vec![0, 1, 2],
            PartitionWeights::NonUniform(vec![1.0, 0.5, 0.25]),
        )
        .expect("matching non-uniform weights"),
        vec![4.0, 8.0, 16.0],
        vec![4.0, 4.0, 4.0],
        vec![4.0, 2.0, 1.0]
    )]
    fn test_restrict_and_weight(
        #[case] core: SubdomainCore,
        #[case] global: Vec<f64>,
        #[case] restricted: Vec<f64>,
        #[case] weighted: Vec<f64>,
    ) {
        let mut local = vec![0.0; restricted.len()];
        core.restrict_weighted(&global, &mut local);
        assert_eq!(local, restricted);

        core.weight(&mut local);
        assert_eq!(local, weighted);
    }

    #[test]
    fn test_apply_weighted_with_identity_solver() {
        let core = SubdomainCore::uniform(vec![1, 2]);
        let solver = IdentityLocalSolver { n: 2 };
        let entry = SubdomainEntry::try_new(core, solver).expect("valid entry");

        let r = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 2];
        entry
            .apply_weighted_local(&r, &mut out, Threads::All, MemStack::new(&mut []))
            .expect("identity local solver should not fail");

        // Restricts DOFs [1, 2] → [20, 30] and solves with the identity.
        assert_eq!(out, vec![20.0, 30.0]);
    }
}
