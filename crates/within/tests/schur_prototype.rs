#![cfg(feature = "schur-prototype")]

use ndarray::Array2;
use within::{Effect, LocalSolverConfig, LsmrOptions, PreconditionerConfig, Solver};

#[test]
fn reduced_solve_matches_full_fit_and_normal_equations() {
    let options = LsmrOptions {
        tol: 1e-10,
        maxiter: 1000,
        local_size: Some(20),
    };
    for sizes in [[60, 20, 10], [10, 60, 20], [20, 10, 60], [30, 30, 30]] {
        for disconnected in [false, true] {
            let n = 2400;
            let mut state = 41u64;
            let mut categories = Array2::zeros((n, 3));
            for i in 0..n {
                for j in 0..3 {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let half = sizes[j] / 2;
                    categories[[i, j]] = if disconnected {
                        ((state >> 32) as usize % half + (i % 2) * half) as u32
                    } else {
                        ((state >> 32) as usize % sizes[j]) as u32
                    };
                }
            }
            let y: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.71).sin() + (categories[[i, 0]] as f64).cos())
                .collect();
            let weights: Vec<f64> = (0..n)
                .map(|i| {
                    if categories[[i, 1]] == 0 {
                        0.0
                    } else {
                        0.2 + (i % 11) as f64
                    }
                })
                .collect();
            for weighted in [false, true] {
                for config in [
                    PreconditionerConfig::Off,
                    PreconditionerConfig::Diagonal,
                    PreconditionerConfig::Additive {
                        local_solver: LocalSolverConfig::default(),
                        reduction: Default::default(),
                    },
                ] {
                    let solver = Solver::new(
                        categories.view(),
                        weighted.then_some(weights.as_slice()),
                        config.clone(),
                    )
                    .unwrap();
                    let full = solver.solve(&y, &options).unwrap();
                    let reduced = solver.solve_schur_prototype(&y, &options).unwrap();
                    assert!(full.converged && reduced.converged, "{sizes:?} disconnected={disconnected} weighted={weighted} {config:?}: {} {}", full.residual, reduced.residual);
                    let mut fit_error = 0.0;
                    let mut gradient = vec![0.0; sizes.iter().sum()];
                    let mut reference = gradient.clone();
                    for i in 0..n {
                        let w = if weighted { weights[i] } else { 1.0 };
                        fit_error += w * (full.demeaned[i] - reduced.demeaned[i]).powi(2);
                        let mut offset = 0;
                        for j in 0..3 {
                            gradient[offset + categories[[i, j]] as usize] +=
                                w * reduced.demeaned[i];
                            reference[offset + categories[[i, j]] as usize] += w * y[i];
                            offset += sizes[j];
                        }
                    }
                    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    assert!(fit_error.sqrt() < 1e-6, "fit error {}", fit_error.sqrt());
                    assert!(norm(&gradient) / norm(&reference) < 1e-8);
                    assert!(reduced
                        .x
                        .iter()
                        .chain(&reduced.demeaned)
                        .all(|x| x.is_finite()));
                }
            }
        }
    }
}

#[test]
fn handles_zero_rhs_single_factor_and_rejects_slopes() {
    let categories = Array2::from_shape_fn((100, 2), |(i, j)| ((i / (j + 1)) % 7) as u32);
    let solver = Solver::new(categories.view(), None, PreconditionerConfig::Diagonal).unwrap();
    let options = LsmrOptions::default();
    let zero = solver
        .solve_schur_prototype(&vec![0.0; 100], &options)
        .unwrap();
    assert!(zero.converged);
    assert!(zero.demeaned.iter().all(|&x| x == 0.0));
    assert!(solver.solve_schur_prototype(&[0.0], &options).is_err());
    let single = categories.slice_axis(ndarray::Axis(1), ndarray::Slice::from(..1));
    let solver = Solver::new(single, None, PreconditionerConfig::Diagonal).unwrap();
    let y: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
    let result = solver.solve_schur_prototype(&y, &options).unwrap();
    assert!(result.converged);
    let levels = vec![0; 100];
    let effect = Effect::new(&levels, true, [&y[..]]).unwrap();
    let solver = Solver::new(vec![effect], None, PreconditionerConfig::Diagonal).unwrap();
    assert!(solver.solve_schur_prototype(&y, &options).is_err());
}
