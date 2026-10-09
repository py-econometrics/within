use rayon::prelude::*;
use schwarz_precond::{
    LocalSolveError, LocalSolver, PartitionWeights, ReductionStrategy, SchwarzPreconditioner,
    SubdomainCore, SubdomainEntry,
};

struct Diagonal {
    n: usize,
    divisor: f64,
}
impl LocalSolver for Diagonal {
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
        _: bool,
    ) -> Result<(), LocalSolveError> {
        for (out, &value) in sol.iter_mut().zip(rhs.iter()) {
            *out = value / self.divisor;
        }
        Ok(())
    }
}

#[test]
fn overlapping_reduction_matches_serial_subdomain_order_even_under_concurrent_reuse() {
    let n = 257;
    let entries: Vec<_> = (0..37)
        .map(|s| {
            let core = SubdomainCore::with_partition_weights(
                (0..n as u32).collect(),
                PartitionWeights::NonUniform(
                    (0..n)
                        .map(|i| 0.1 + ((i + 7 * s) % 23) as f64 / 17.)
                        .collect(),
                ),
            )
            .unwrap();
            SubdomainEntry::try_new(
                core,
                Diagonal {
                    n,
                    divisor: 1. + s as f64 / 13.,
                },
            )
            .unwrap()
        })
        .collect();
    let rhs: Vec<_> = (0..n)
        .map(|i| (i as f64 * 0.19).sin() * (1. + (i % 13) as f64))
        .collect();
    let mut expected = vec![0.; n + 1];
    for entry in &entries {
        entry
            .apply_weighted_into_with_scratch(
                &rhs,
                &mut expected,
                &mut vec![0.; n],
                &mut vec![0.; n],
                false,
            )
            .unwrap();
    }
    let mut rhs = rhs;
    rhs.push(7.); // An uncovered coordinate must remain zero.
    let preconditioner =
        SchwarzPreconditioner::with_n_dofs(entries, n + 1, ReductionStrategy::ParallelReduction);
    let expected: Vec<_> = expected.iter().map(|v| v.to_bits()).collect();
    for threads in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            (0..16).into_par_iter().for_each(|_| {
                let clone = preconditioner.clone();
                let mut output = vec![f64::NAN; n + 1];
                for _ in 0..3 {
                    clone.apply(&rhs, &mut output).unwrap();
                    assert!(
                        output
                            .iter()
                            .map(|v| v.to_bits())
                            .eq(expected.iter().copied()),
                        "threads={threads}"
                    );
                }
            });
        });
    }
}

#[test]
fn failed_reduction_clears_output_and_can_be_retried() {
    use schwarz_precond::SolveError;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct FailOnce(AtomicBool);
    impl LocalSolver for FailOnce {
        fn n_local(&self) -> usize {
            2
        }
        fn scratch_size(&self) -> usize {
            2
        }
        fn solve_local(
            &self,
            rhs: &mut [f64],
            sol: &mut [f64],
            _: bool,
        ) -> Result<(), LocalSolveError> {
            if self.0.swap(false, Ordering::Relaxed) {
                return Err(LocalSolveError::BackendFailed {
                    context: "test.fail_once",
                    message: "deliberate failure".into(),
                });
            }
            sol.copy_from_slice(rhs);
            Ok(())
        }
    }
    for workers in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let entry = SubdomainEntry::try_new(
                SubdomainCore::uniform(vec![0, 1]),
                FailOnce(AtomicBool::new(true)),
            )
            .unwrap();
            let preconditioner =
                SchwarzPreconditioner::new(vec![entry], ReductionStrategy::ParallelReduction);
            let mut output = vec![f64::NAN; 2];
            assert!(matches!(
                preconditioner.apply(&[1., 2.], &mut output),
                Err(SolveError::LocalSolveFailed { subdomain: 0, .. })
            ));
            assert_eq!(output, [0., 0.]);
            preconditioner.apply(&[1., 2.], &mut output).unwrap();
            assert_eq!(output, [1., 2.]);
        });
    }
}
