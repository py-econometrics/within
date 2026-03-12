use std::sync::Mutex;

use crate::domain::PartitionWeights;
use crate::error::{validate_entries, ApplyError, LocalSolveError, PreconditionerBuildError};
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::Operator;

/// Trait for updating the global residual after a single subdomain correction
/// in a multiplicative Schwarz sweep.
///
/// The multiplicative Schwarz preconditioner needs to update `r_work` after each
/// subdomain solve. Different strategies exist:
/// - `OperatorResidualUpdater`: naive full recompute via `r = b - A * y_accum`
/// - Observation-space updater (in `within` crate): exploits FE structure for sparse updates
pub trait ResidualUpdater: Send + Sync {
    /// Update the working residual after a subdomain correction.
    ///
    /// `global_indices`: the DOF indices touched by this subdomain
    /// `weighted_correction`: the PoU-weighted local correction (same length as `global_indices`)
    /// `r_work`: the full global residual vector to update in-place
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]);

    /// Reset before a new sweep. Called at the start of each forward/backward sweep.
    fn reset(&mut self, r_original: &[f64]);
}

/// Naive residual updater that recomputes `r = b - A * y_accum` after each subdomain.
///
/// Maintains an internal accumulator `y_accum`. On each `update()`:
/// 1. Scatter `weighted_correction` into `y_accum` at `global_indices`
/// 2. Recompute `r_work = r_original - A * y_accum`
///
/// This is `O(n_dofs)` per update — correct but slow. Useful as a testing baseline.
pub struct OperatorResidualUpdater<'a, A: Operator> {
    operator: &'a A,
    /// Accumulator: sum of all corrections applied so far
    y_accum: Vec<f64>,
    /// Original residual (r before the sweep started)
    r_original: Vec<f64>,
    /// Scratch buffer for A * y_accum
    a_y: Vec<f64>,
}

impl<'a, A: Operator> OperatorResidualUpdater<'a, A> {
    /// Create a new updater.
    pub fn new(operator: &'a A, n_dofs: usize) -> Self {
        Self {
            operator,
            y_accum: vec![0.0; n_dofs],
            r_original: vec![0.0; n_dofs],
            a_y: vec![0.0; n_dofs],
        }
    }
}

impl<A: Operator> ResidualUpdater for OperatorResidualUpdater<'_, A> {
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]) {
        for (k, &gi) in global_indices.iter().enumerate() {
            self.y_accum[gi as usize] += weighted_correction[k];
        }

        self.operator.apply(&self.y_accum, &mut self.a_y);
        for (r, (&ro, &ay)) in r_work
            .iter_mut()
            .zip(self.r_original.iter().zip(self.a_y.iter()))
        {
            *r = ro - ay;
        }
    }

    fn reset(&mut self, r_original: &[f64]) {
        self.r_original.copy_from_slice(r_original);
        self.y_accum.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Scratch buffers for a multiplicative sweep (single-threaded).
struct SweepBuffers {
    /// Working copy of the residual, updated after each subdomain solve.
    r_work: Vec<f64>,
    /// Local RHS scratch (length = max_scratch_size).
    r_scratch: Vec<f64>,
    /// Local solution scratch (length = max_scratch_size).
    z_scratch: Vec<f64>,
    /// Partition-of-unity-weighted local solution (length = max local DOFs).
    weighted_local_sol: Vec<f64>,
}

impl SweepBuffers {
    fn new(n_dofs: usize, max_scratch_size: usize, max_local_dofs: usize) -> Self {
        Self {
            r_work: vec![0.0; n_dofs],
            r_scratch: vec![0.0; max_scratch_size],
            z_scratch: vec![0.0; max_scratch_size],
            weighted_local_sol: vec![0.0; max_local_dofs],
        }
    }
}

fn apply_subdomain<S: LocalSolver, U: ResidualUpdater>(
    entry: &SubdomainEntry<S>,
    bufs: &mut SweepBuffers,
    z: &mut [f64],
    updater: &mut U,
) -> Result<(), LocalSolveError> {
    if entry.is_empty() {
        return Ok(());
    }

    let n_local = entry.global_indices().len();

    entry
        .core()
        .restrict_weighted(&bufs.r_work, &mut bufs.r_scratch);
    entry
        .solver()
        .solve_local(&mut bufs.r_scratch, &mut bufs.z_scratch)?;
    entry.core().prolongate_weighted_add(&bufs.z_scratch, z);

    match entry.partition_weights() {
        PartitionWeights::Uniform(_) => {
            updater.update(
                entry.global_indices(),
                &bufs.z_scratch[..n_local],
                &mut bufs.r_work,
            );
        }
        PartitionWeights::NonUniform(w) => {
            for (k, wk) in w.iter().enumerate().take(n_local) {
                bufs.weighted_local_sol[k] = wk * bufs.z_scratch[k];
            }
            updater.update(
                entry.global_indices(),
                &bufs.weighted_local_sol[..n_local],
                &mut bufs.r_work,
            );
        }
    }
    Ok(())
}

fn compute_sizes<S: LocalSolver>(entries: &[SubdomainEntry<S>]) -> (usize, usize) {
    let max_scratch_size = entries.iter().map(|e| e.scratch_size()).max().unwrap_or(0);
    let max_local_dofs = entries
        .iter()
        .map(|e| e.global_indices().len())
        .max()
        .unwrap_or(0);
    (max_scratch_size, max_local_dofs)
}

/// Multiplicative Schwarz preconditioner, generic over local solver and residual updater.
pub struct MultiplicativeSchwarzPreconditioner<S: LocalSolver, U: ResidualUpdater> {
    subdomains: Vec<SubdomainEntry<S>>,
    updater: Mutex<U>,
    n_dofs: usize,
    symmetric: bool,
    scratch: Mutex<SweepBuffers>,
}

impl<S: LocalSolver, U: ResidualUpdater> MultiplicativeSchwarzPreconditioner<S, U> {
    /// Construct a multiplicative Schwarz preconditioner.
    pub fn new(
        entries: Vec<SubdomainEntry<S>>,
        updater: U,
        n_dofs: usize,
        symmetric: bool,
    ) -> Result<Self, PreconditionerBuildError> {
        validate_entries(&entries, n_dofs)?;
        let (max_scratch_size, max_local_dofs) = compute_sizes(&entries);
        Ok(Self {
            subdomains: entries,
            updater: Mutex::new(updater),
            n_dofs,
            symmetric,
            scratch: Mutex::new(SweepBuffers::new(n_dofs, max_scratch_size, max_local_dofs)),
        })
    }

    /// Access the underlying subdomain entries.
    pub fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    /// Fallible operator apply that propagates local-solver failures.
    pub fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        let mut bufs = self
            .scratch
            .lock()
            .map_err(|_| ApplyError::Synchronization {
                context: "multiplicative.scratch.lock",
            })?;
        let mut updater = self
            .updater
            .lock()
            .map_err(|_| ApplyError::Synchronization {
                context: "multiplicative.updater.lock",
            })?;
        z.fill(0.0);

        bufs.r_work.copy_from_slice(r);
        updater.reset(r);
        for (subdomain, entry) in self.subdomains.iter().enumerate() {
            apply_subdomain(entry, &mut bufs, z, &mut *updater)
                .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })?;
        }

        if self.symmetric {
            updater.reset(&bufs.r_work);
            for (rev_idx, entry) in self.subdomains.iter().rev().enumerate() {
                let subdomain = self.subdomains.len().saturating_sub(1) - rev_idx;
                apply_subdomain(entry, &mut bufs, z, &mut *updater)
                    .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })?;
            }
        }
        Ok(())
    }
}

impl<S: LocalSolver, U: ResidualUpdater> Operator for MultiplicativeSchwarzPreconditioner<S, U> {
    fn nrows(&self) -> usize {
        self.n_dofs
    }

    fn ncols(&self) -> usize {
        self.n_dofs
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        if self.try_apply(r, z).is_err() {
            z.fill(f64::NAN);
        }
    }

    fn apply_adjoint(&self, r: &[f64], z: &mut [f64]) {
        self.apply(r, z);
    }

    fn try_apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        MultiplicativeSchwarzPreconditioner::try_apply(self, r, z)
    }

    fn try_apply_adjoint(&self, r: &[f64], z: &mut [f64]) -> Result<(), ApplyError> {
        MultiplicativeSchwarzPreconditioner::try_apply(self, r, z)
    }
}

impl<S: LocalSolver + Clone, U: ResidualUpdater + Clone> Clone
    for MultiplicativeSchwarzPreconditioner<S, U>
{
    fn clone(&self) -> Self {
        let updater = self.updater.lock().expect("updater lock poisoned").clone();
        let (max_scratch_size, max_local_dofs) = compute_sizes(&self.subdomains);
        Self {
            subdomains: self.subdomains.clone(),
            updater: Mutex::new(updater),
            n_dofs: self.n_dofs,
            symmetric: self.symmetric,
            scratch: Mutex::new(SweepBuffers::new(
                self.n_dofs,
                max_scratch_size,
                max_local_dofs,
            )),
        }
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use std::sync::Mutex;

    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{
        compute_sizes, MultiplicativeSchwarzPreconditioner, ResidualUpdater, SweepBuffers,
    };
    use crate::local_solve::{LocalSolver, SubdomainEntry};

    impl<S: LocalSolver + Serialize, U: ResidualUpdater + Serialize> Serialize
        for MultiplicativeSchwarzPreconditioner<S, U>
    {
        fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
            let mut state =
                serializer.serialize_struct("MultiplicativeSchwarzPreconditioner", 4)?;
            state.serialize_field("subdomains", &self.subdomains)?;
            let updater = self.updater.lock().map_err(serde::ser::Error::custom)?;
            state.serialize_field("updater", &*updater)?;
            state.serialize_field("n_dofs", &self.n_dofs)?;
            state.serialize_field("symmetric", &self.symmetric)?;
            state.end()
        }
    }

    impl<
            'de,
            S: LocalSolver + serde::de::DeserializeOwned,
            U: ResidualUpdater + serde::de::DeserializeOwned,
        > Deserialize<'de> for MultiplicativeSchwarzPreconditioner<S, U>
    {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            #[serde(bound(
                deserialize = "S: serde::de::DeserializeOwned, U: serde::de::DeserializeOwned"
            ))]
            struct Helper<S: LocalSolver, U: ResidualUpdater> {
                subdomains: Vec<SubdomainEntry<S>>,
                updater: U,
                n_dofs: usize,
                symmetric: bool,
            }

            let h = Helper::deserialize(deserializer)?;
            let (max_scratch_size, max_local_dofs) = compute_sizes(&h.subdomains);
            Ok(MultiplicativeSchwarzPreconditioner {
                scratch: Mutex::new(SweepBuffers::new(
                    h.n_dofs,
                    max_scratch_size,
                    max_local_dofs,
                )),
                subdomains: h.subdomains,
                updater: Mutex::new(h.updater),
                n_dofs: h.n_dofs,
                symmetric: h.symmetric,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DiagOperator {
        values: Vec<f64>,
    }

    impl DiagOperator {
        fn new(values: Vec<f64>) -> Self {
            Self { values }
        }
    }

    impl Operator for DiagOperator {
        fn nrows(&self) -> usize {
            self.values.len()
        }

        fn ncols(&self) -> usize {
            self.values.len()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) {
            for i in 0..self.values.len() {
                y[i] = self.values[i] * x[i];
            }
        }

        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
            self.apply(x, y);
        }
    }

    #[test]
    fn test_operator_residual_updater_basic() {
        let a = DiagOperator::new(vec![2.0, 3.0, 1.0, 4.0]);
        let mut updater = OperatorResidualUpdater::new(&a, 4);

        let r_original = [10.0, 12.0, 5.0, 8.0];
        updater.reset(&r_original);

        let mut r_work = r_original.to_vec();

        updater.update(&[0, 2], &[1.0, 2.0], &mut r_work);
        assert!((r_work[0] - 8.0).abs() < 1e-12);
        assert!((r_work[1] - 12.0).abs() < 1e-12);
        assert!((r_work[2] - 3.0).abs() < 1e-12);
        assert!((r_work[3] - 8.0).abs() < 1e-12);

        updater.update(&[1, 3], &[0.5, 1.0], &mut r_work);
        assert!((r_work[0] - 8.0).abs() < 1e-12);
        assert!((r_work[1] - 10.5).abs() < 1e-12);
        assert!((r_work[2] - 3.0).abs() < 1e-12);
        assert!((r_work[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_operator_residual_updater_reset() {
        let a = DiagOperator::new(vec![1.0, 1.0]);
        let mut updater = OperatorResidualUpdater::new(&a, 2);

        updater.reset(&[5.0, 3.0]);
        let mut r = vec![5.0, 3.0];
        updater.update(&[0], &[2.0], &mut r);
        assert!((r[0] - 3.0).abs() < 1e-12);

        updater.reset(&[10.0, 7.0]);
        let mut r = vec![10.0, 7.0];
        updater.update(&[1], &[1.0], &mut r);
        assert!((r[0] - 10.0).abs() < 1e-12);
        assert!((r[1] - 6.0).abs() < 1e-12);
    }
}
