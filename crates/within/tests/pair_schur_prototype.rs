#![cfg(feature = "pair-schur-prototype")]

use ndarray::Array2;
use within::{LocalSolverConfig, LsmrOptions, PreconditionerConfig, SchurMode, Solver};

fn config() -> LocalSolverConfig {
    LocalSolverConfig {
        dense_threshold: 0,
        schur: SchurMode::Exact,
        ..Default::default()
    }
}

#[test]
fn pair_schur_matches_full_fits_on_weighted_and_disconnected_designs() {
    let options = LsmrOptions {
        tol: 1e-9,
        maxiter: 2000,
        local_size: None,
    };
    for sizes in [[40, 12, 6], [12, 40, 6], [12, 6, 40]] {
        for disconnected in [false, true] {
            let n = 1200;
            let mut state = 713u64;
            let categories = Array2::from_shape_fn((n, 3), |(i, j)| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let count = if disconnected { sizes[j] / 2 } else { sizes[j] };
                ((state >> 32) as usize % count + if disconnected { (i % 2) * count } else { 0 })
                    as u32
            });
            let y: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.37).sin() + categories[[i, 0]] as f64)
                .collect();
            let weights: Vec<f64> = (0..n)
                .map(|i| {
                    if categories[[i, 1]] == 0 {
                        0.0
                    } else {
                        0.5 + (i % 7) as f64
                    }
                })
                .collect();
            for weighted in [false, true] {
                let full = Solver::new(
                    categories.view(),
                    weighted.then_some(weights.as_slice()),
                    PreconditionerConfig::Additive {
                        local_solver: config(),
                        reduction: Default::default(),
                    },
                )
                .unwrap();
                let reference = full
                    .solve(
                        &y,
                        &LsmrOptions {
                            tol: 1e-12,
                            ..options.clone()
                        },
                    )
                    .unwrap();
                assert!(reference.converged);
                let reduced = full
                    .pair_schur_prototype(&config(), Default::default())
                    .unwrap();
                let jacobi = full
                    .pair_schur_block_jacobi(&config(), Default::default())
                    .unwrap();
                assert!(
                    jacobi.n_subdomains() > 0 && reduced.n_subdomains() > jacobi.n_subdomains()
                );
                for multiplier in [1.0, -0.5] {
                    let response: Vec<f64> = y.iter().map(|v| multiplier * v).collect();
                    let result = reduced.solve_lsmr(&response, &options).unwrap();
                    assert!(result.converged, "{sizes:?} disconnected={disconnected} weighted={weighted} residual={} iterations={}", result.residual,result.iterations);
                    let mut gradient = vec![0.0; sizes.iter().sum()];
                    let mut rhs = gradient.clone();
                    for i in 0..n {
                        let w = if weighted { weights[i] } else { 1.0 };
                        let mut offset = 0;
                        for j in 0..3 {
                            let k = offset + categories[[i, j]] as usize;
                            gradient[k] += w * result.demeaned[i];
                            rhs[k] += w * response[i];
                            offset += sizes[j];
                        }
                    }
                    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let relative_gradient = norm(&gradient) / norm(&rhs);
                    assert!(
                        relative_gradient <= options.tol,
                        "full gradient {relative_gradient}"
                    );
                    let error = result
                        .demeaned
                        .iter()
                        .zip(&reference.demeaned)
                        .enumerate()
                        .map(|(i, (a, b))| {
                            let w = if weighted { weights[i] } else { 1.0 };
                            w * (a - multiplier * b).powi(2)
                        })
                        .sum::<f64>()
                        .sqrt();
                    assert!(error < 1e-4, "fit error {error}");
                }
                let zero = reduced.solve_lsmr(&vec![0.0; n], &options).unwrap();
                assert!(zero.converged && zero.iterations == 0);
                let x: Vec<f64> = (0..reduced.n_reduced()).map(|i| (i as f64).sin()).collect();
                let z: Vec<f64> = (0..reduced.n_reduced())
                    .map(|i| (i as f64 * 0.7).cos())
                    .collect();
                let mut bx = vec![0.0; x.len()];
                let mut bz = bx.clone();
                reduced.apply_preconditioner(&x, &mut bx).unwrap();
                reduced.apply_preconditioner(&z, &mut bz).unwrap();
                let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
                assert!((dot(&x, &bz) - dot(&z, &bx)).abs() < 1e-9);
                assert!(dot(&x, &bx) >= -1e-12);
            }
        }
    }
}

#[test]
fn global_elimination_can_reverse_local_pair_orientation() {
    let categories = Array2::from_shape_fn((200, 2), |(i, j)| {
        if i < 100 {
            if j == 0 {
                0
            } else {
                (i % 5) as u32
            }
        } else if j == 0 {
            1 + (i % 12) as u32
        } else {
            5
        }
    });
    let solver = Solver::new(categories.view(), None, PreconditionerConfig::Off).unwrap();
    let reduced = solver
        .pair_schur_prototype(&config(), Default::default())
        .unwrap();
    let jacobi = solver
        .pair_schur_block_jacobi(&config(), Default::default())
        .unwrap();
    assert_eq!((reduced.n_subdomains(), jacobi.n_subdomains()), (2, 2));
    let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.31).sin()).collect();
    let options = LsmrOptions {
        tol: 1e-9,
        maxiter: 1000,
        local_size: None,
    };
    let r = reduced.solve_lsmr(&y, &options).unwrap();
    let reference = solver.solve(&y, &options).unwrap();
    assert!(r.converged && reference.converged);
    assert!(
        r.demeaned
            .iter()
            .zip(&reference.demeaned)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            < 1e-6
    );
}

#[test]
fn all_zero_weights_and_invalid_inputs() {
    let categories = Array2::from_shape_fn((100, 2), |(i, j)| ((i / (j + 1)) % 9) as u32);
    let solver = Solver::new(
        categories.view(),
        Some(&vec![0.0; 100]),
        PreconditionerConfig::Off,
    )
    .unwrap();
    let reduced = solver
        .pair_schur_prototype(&config(), Default::default())
        .unwrap();
    let y = vec![1.0; 100];
    let r = reduced.solve_lsmr(&y, &LsmrOptions::default()).unwrap();
    assert!(r.converged);
    assert_eq!(r.demeaned, y);
    assert!(reduced.solve_lsmr(&[1.0], &LsmrOptions::default()).is_err());
}
