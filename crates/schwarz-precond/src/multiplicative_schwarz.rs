//! Multiplicative Schwarz preconditioner: sequential subdomain sweeps with
//! residual updates between subdomains.

use std::sync::Mutex;

use crate::domain::PartitionWeights;
use crate::error::{validate_entries, ApplyError, LocalSolveError, PreconditionerBuildError};
use crate::local_solve::{LocalSolver, SubdomainEntry};
use crate::residual_update::ResidualUpdater;
use crate::Operator;

/// Scratch buffers for a multiplicative sweep (single-threaded).
struct SweepBuffers {
    /// Working copy of the residual, updated after each subdomain solve.
    r_work: Vec<f64>,
    /// Local RHS scratch (length = max_scratch_size).
    r_scratch: Vec<f64>,
    /// Local solution scratch (length = max_scratch_size).
    z_scratch: Vec<f64>,
    /// Weighted correction for residual update (length = max local DOFs).
    correction: Vec<f64>,
}

impl SweepBuffers {
    fn new(n_dofs: usize, max_scratch_size: usize, max_local_dofs: usize) -> Self {
        Self {
            r_work: vec![0.0; n_dofs],
            r_scratch: vec![0.0; max_scratch_size],
            z_scratch: vec![0.0; max_scratch_size],
            correction: vec![0.0; max_local_dofs],
        }
    }
}

/// Process a single subdomain: restrict, local solve, prolongate, update residual.
fn apply_subdomain<S: LocalSolver, U: ResidualUpdater>(
    entry: &SubdomainEntry<S>,
    r_work: &mut [f64],
    z: &mut [f64],
    r_scratch: &mut [f64],
    z_scratch: &mut [f64],
    correction: &mut [f64],
    updater: &mut U,
) -> Result<(), LocalSolveError> {
    if entry.core.global_indices.is_empty() {
        return Ok(());
    }

    let n_local = entry.core.global_indices.len();

    entry.core.restrict_weighted(r_work, r_scratch);
    entry.solver.solve_local(r_scratch, z_scratch)?;
    entry.core.prolongate_weighted_add(z_scratch, z);

    match &entry.core.partition_weights {
        PartitionWeights::Uniform(_) => {
            updater.update(&entry.core.global_indices, &z_scratch[..n_local], r_work);
        }
        PartitionWeights::NonUniform(w) => {
            for k in 0..n_local {
                correction[k] = w[k] * z_scratch[k];
            }
            updater.update(&entry.core.global_indices, &correction[..n_local], r_work);
        }
    }
    Ok(())
}

/// Compute subdomain sizing metadata.
fn compute_sizes<S: LocalSolver>(entries: &[SubdomainEntry<S>]) -> (usize, usize) {
    let max_scratch_size = entries.iter().map(|e| e.scratch_size()).max().unwrap_or(0);
    let max_local_dofs = entries
        .iter()
        .map(|e| e.core.global_indices.len())
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
        let SweepBuffers {
            ref mut r_work,
            ref mut r_scratch,
            ref mut z_scratch,
            ref mut correction,
        } = *bufs;

        z.fill(0.0);

        r_work.copy_from_slice(r);
        updater.reset(r);
        for (subdomain, entry) in self.subdomains.iter().enumerate() {
            apply_subdomain(
                entry,
                r_work,
                z,
                r_scratch,
                z_scratch,
                correction,
                &mut *updater,
            )
            .map_err(|source| ApplyError::LocalSolveFailed { subdomain, source })?;
        }

        if self.symmetric {
            updater.reset(r_work);
            for (rev_idx, entry) in self.subdomains.iter().rev().enumerate() {
                let subdomain = self.subdomains.len().saturating_sub(1) - rev_idx;
                apply_subdomain(
                    entry,
                    r_work,
                    z,
                    r_scratch,
                    z_scratch,
                    correction,
                    &mut *updater,
                )
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
