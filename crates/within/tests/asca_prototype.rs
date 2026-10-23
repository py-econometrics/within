#![cfg(feature = "asca-prototype")]

use ndarray::Array2;
use std::num::NonZeroUsize;
use within::{AscaConfig, Effect, LsmrOptions, PreconditionerConfig, Solver};

#[test]
fn assembled_and_sampled_solvers_match_full_weighted_fits() {
    let options = LsmrOptions {
        tol: 1e-9,
        maxiter: 2000,
        local_size: None,
    };
    for sizes in [[40, 12, 6], [12, 40, 6], [12, 6, 40]] {
        for disconnected in [false, true] {
            let n = 1200;
            let mut state = 731u64;
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
                let solver = Solver::new(
                    categories.view(),
                    weighted.then_some(weights.as_slice()),
                    PreconditionerConfig::Diagonal,
                )
                .unwrap();
                let reference = solver
                    .solve(
                        &y,
                        &LsmrOptions {
                            tol: 1e-12,
                            ..options.clone()
                        },
                    )
                    .unwrap();
                assert!(reference.converged);
                for pairs in [None, NonZeroUsize::new(8), NonZeroUsize::new(32)] {
                    let asca = solver
                        .asca_prototype(AscaConfig {
                            pairs_per_level: pairs,
                            ..Default::default()
                        })
                        .unwrap();
                    for multiplier in [1.0, -0.5] {
                        let response: Vec<f64> = y.iter().map(|v| multiplier * v).collect();
                        let result = asca.solve(&response, &options).unwrap();
                        assert!(result.converged, "{sizes:?} disconnected={disconnected} weighted={weighted} pairs={pairs:?} iterations={} residual={}", result.iterations, result.residual);
                        assert!(result.residual <= options.tol);
                        let error: f64 = result
                            .demeaned
                            .iter()
                            .zip(&reference.demeaned)
                            .enumerate()
                            .map(|(i, (a, b))| {
                                let w = if weighted { weights[i] } else { 1.0 };
                                w * (a - multiplier * b).powi(2)
                            })
                            .sum();
                        assert!(error.sqrt() < 1e-4, "fit error {}", error.sqrt());
                        assert!(result
                            .x
                            .iter()
                            .chain(&result.demeaned)
                            .all(|v| v.is_finite()));
                    }
                    let zero = asca.solve(&vec![0.0; n], &options).unwrap();
                    assert!(zero.converged && zero.iterations == 0);
                }
            }
        }
    }
}

#[test]
fn validates_input_and_handles_zero_weight_design() {
    let categories = Array2::from_shape_fn((100, 2), |(i, j)| ((i / (j + 1)) % 9) as u32);
    let solver = Solver::new(
        categories.view(),
        Some(&vec![0.0; 100]),
        PreconditionerConfig::Off,
    )
    .unwrap();
    for relative_shift in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(solver
            .asca_prototype(AscaConfig {
                relative_shift,
                ..Default::default()
            })
            .is_err());
    }
    let asca = solver.asca_prototype(AscaConfig::default()).unwrap();
    let y: Vec<f64> = (0..100).map(|i| (i as f64).cos()).collect();
    let result = asca.solve(&y, &LsmrOptions::default()).unwrap();
    assert!(result.converged);
    assert_eq!(result.demeaned, y);
    assert!(asca.solve(&[1.0], &LsmrOptions::default()).is_err());
    let levels = vec![0; 100];
    let effect = Effect::new(&levels, true, [&y[..]]).unwrap();
    let slopes = Solver::new(vec![effect], None, PreconditionerConfig::Off).unwrap();
    assert!(slopes.asca_prototype(AscaConfig::default()).is_err());
}

#[test]
fn restricted_schwarz_control_uses_the_same_exact_system() {
    let categories = Array2::from_shape_fn((600, 3), |(i, j)| {
        ((i * (j + 1) + i / 11) % [60, 17, 7][j]) as u32
    });
    let y: Vec<f64> = (0..600).map(|i| (i as f64 * 0.71).cos()).collect();
    let solver = Solver::new(
        categories.view(),
        None,
        PreconditionerConfig::Additive {
            local_solver: Default::default(),
            reduction: Default::default(),
        },
    )
    .unwrap();
    let asca = solver.asca_prototype(AscaConfig::default()).unwrap();
    let options = LsmrOptions {
        tol: 1e-9,
        maxiter: 1000,
        local_size: None,
    };
    let assembled = asca.solve(&y, &options).unwrap();
    let restricted = asca
        .solve_with_full_preconditioner(&y, &options, solver.preconditioner().unwrap())
        .unwrap();
    assert!(assembled.converged && restricted.converged);
    let error = assembled
        .demeaned
        .iter()
        .zip(&restricted.demeaned)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(error < 1e-7);
    let other_categories = Array2::zeros((20, 2));
    let other = Solver::new(
        other_categories.view(),
        None,
        PreconditionerConfig::Diagonal,
    )
    .unwrap();
    assert!(asca
        .solve_with_full_preconditioner(&y, &options, other.preconditioner().unwrap())
        .is_err());
}
