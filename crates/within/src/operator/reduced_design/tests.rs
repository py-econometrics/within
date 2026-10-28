use std::sync::Mutex;

use super::*;
use crate::domain::PreparedDesign;
use crate::{Design, Effect, IntoDesign};
use ndarray::Array2;

fn scale(e: &Elimination<'_>, row: usize) -> f64 {
    e.prepared().sqrt_weights().map_or(1.0, |w| w[row])
}

/// The serial eliminated fit to `values`, laid out like `removed`.
fn fit_coefficients(e: &Elimination<'_>, values: &[f64], coefficients: &mut [f64]) {
    let (n, c) = (e.rows().n_levels(), e.width());
    let levels = e.prepared().design.terms[e.term()].levels();
    let loadings = e.eliminated_loadings();
    let mut sums = vec![0.0; n * c];
    for (i, &level) in levels.iter().enumerate() {
        let v = scale(e, i) * values[i];
        for (sum, a) in sums[level as usize * c..][..c].iter_mut().zip(&loadings) {
            *sum += a.map_or(v, |z| z[i] * v);
        }
    }
    for (l, sums) in sums.chunks_exact(c).enumerate() {
        let inverse = &e.inverse()[l * c * c..][..c * c];
        for (j, row) in inverse.chunks_exact(c).enumerate() {
            coefficients[j * n + l] = row.iter().zip(sums).map(|(g, s)| g * s).sum();
        }
    }
}

/// The serial `(I − P_e)` in original row order.
fn project(e: &Elimination<'_>, values: &mut [f64], fit: &mut [f64]) {
    fit_coefficients(e, values, fit);
    let n = e.rows().n_levels();
    let levels = e.prepared().design.terms[e.term()].levels();
    let loadings = e.eliminated_loadings();
    for (i, &level) in levels.iter().enumerate() {
        let fitted: f64 = (loadings.iter().enumerate())
            .map(|(j, a)| {
                let b = fit[j * n + level as usize];
                a.map_or(b, |z| z[i] * b)
            })
            .sum();
        values[i] -= scale(e, i) * fitted;
    }
}

struct ProjectionScratch {
    observations: Vec<f64>,
    means: Vec<f64>,
}

/// The serial reference [`SegmentedDesign`] is checked against.
struct ProjectedDesign<'a> {
    elimination: &'a Elimination<'a>,
    scratch: Mutex<ProjectionScratch>,
}

impl<'a> ProjectedDesign<'a> {
    fn new(elimination: &'a Elimination<'a>) -> Self {
        Self {
            elimination,
            scratch: Mutex::new(ProjectionScratch {
                observations: vec![0.0; elimination.prepared().design.n_obs],
                means: vec![0.0; elimination.removed().len()],
            }),
        }
    }
}

impl Operator for ProjectedDesign<'_> {
    fn nrows(&self) -> usize {
        self.elimination.prepared().design.n_obs
    }
    fn ncols(&self) -> usize {
        self.elimination.n_kept()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let e = self.elimination;
        y.fill(0.0);
        for channel in e.kept_channels() {
            for (i, &level) in channel.levels.iter().enumerate() {
                let x = x[channel.start + level as usize];
                y[i] += channel.loading.map_or(x, |z| z[i] * x);
            }
        }
        for (i, yi) in y.iter_mut().enumerate() {
            *yi *= scale(e, i);
        }
        let mut scratch = self.scratch.lock().expect("projection scratch poisoned");
        project(e, y, &mut scratch.means);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let e = self.elimination;
        let mut scratch = self.scratch.lock().expect("projection scratch poisoned");
        let ProjectionScratch {
            observations,
            means,
        } = &mut *scratch;
        observations.copy_from_slice(x);
        project(e, observations, means);
        y.fill(0.0);
        for channel in e.kept_channels() {
            for (i, &level) in channel.levels.iter().enumerate() {
                let v = scale(e, i) * observations[i];
                y[channel.start + level as usize] += channel.loading.map_or(v, |z| z[i] * v);
            }
        }
        Ok(())
    }
}

#[test]
#[ignore]
fn profile_reduced_operator() {
    use std::{hint::black_box, time::Instant};
    let n = 1_000_000;
    let channels = std::env::var("BENCH_REDUCED_CHANNELS")
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(2);
    let shuffled = std::env::var_os("BENCH_REDUCED_SHUFFLE").is_some();
    let unsorted = std::env::var_os("BENCH_REDUCED_UNSORTED").is_some();
    let weights = std::env::var_os("BENCH_REDUCED_WEIGHTED").map(|_| weights(n));
    for retained in [4_000, 100_000] {
        let categories = Array2::from_shape_fn((n, channels + 1), |(i, j)| {
            let i = if shuffled { i * 7919 % n } else { i };
            (match j {
                0 => i / 5,
                1 => (i * 7919 + i / 13) % retained,
                _ => (i / (j - 1)) % 10,
            }) as u32
        });
        let columns: Vec<Vec<u32>> = categories
            .columns()
            .into_iter()
            .map(|c| c.to_vec())
            .collect();
        let design = if unsorted {
            Design::new_unsorted(columns.iter().map(|c| Effect::new(c, true, []).unwrap())).unwrap()
        } else {
            Design::from_categories(categories.view()).unwrap()
        };
        let prepared =
            PreparedDesign::new(design.into_design().unwrap(), weights.as_deref()).unwrap();
        let mut setups = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            let e = Elimination::new(&prepared);
            black_box(build(&e));
            setups.push(start.elapsed().as_secs_f64());
        }
        setups.sort_by(f64::total_cmp);
        let e = Elimination::new(&prepared);
        let op = build(&e);
        let mut x: Vec<f64> = (0..op.ncols()).map(|i| (i as f64).sin()).collect();
        let mut y = vec![0.0; n];
        let mut times = [0.0; 3];
        for (adjoint, seconds) in times.iter_mut().enumerate() {
            let start = Instant::now();
            for _ in 0..100 {
                if adjoint == 0 {
                    op.apply(black_box(&x), black_box(&mut y)).unwrap();
                } else if adjoint == 1 {
                    op.apply_adjoint(black_box(&y), black_box(&mut x)).unwrap();
                } else {
                    black_box(op.projected_rhs(black_box(&y)));
                }
            }
            *seconds = start.elapsed().as_secs_f64() / 100.0;
        }
        println!(
            "retained={retained} setup={:.6} forward={:.6} adjoint={:.6} rhs={:.6}",
            setups[3], times[0], times[1], times[2]
        );
    }
}

fn weights(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i % 7 == 0 {
                0.0
            } else {
                0.2 + (i % 3) as f64
            }
        })
        .collect()
}

#[test]
fn adjoint_overwrites_output_across_pools_and_concurrent_calls() {
    let n = 100_000;
    let categories = Array2::from_shape_fn((n, 3), |(i, j)| {
        [(i * 7919 + i / 13) % 20_000, i % 10, i / 2][j] as u32
    });
    let weights: Vec<f64> = (0..n)
        .map(|i| {
            if i / 2 % 7 == 0 {
                0.0
            } else {
                0.2 + (i % 3) as f64
            }
        })
        .collect();
    let prepared =
        PreparedDesign::new(categories.view().into_design().unwrap(), Some(&weights)).unwrap();
    let e = Elimination::new(&prepared);
    let reference = ProjectedDesign::new(&e);
    let op = build(&e);
    let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
    let grouped: Vec<f64> = (e.rows().order.rows(0..n)).map(|i| u[i as usize]).collect();
    let mut expected = vec![0.0; op.ncols()];
    reference.apply_adjoint(&u, &mut expected).unwrap();
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (0..8).into_par_iter().for_each(|k| {
                    let scale = k as f64 - 3.0;
                    let input: Vec<f64> = grouped.iter().map(|v| scale * v).collect();
                    let mut got = vec![f64::NAN; op.ncols()];
                    for _ in 0..3 {
                        op.apply_adjoint(&input, &mut got).unwrap();
                        for (a, b) in got.iter().zip(&expected) {
                            assert!((a - scale * b).abs() < 1e-10 * (1.0 + b.abs()));
                        }
                    }
                });
            });
    }
}

fn assert_reproduces_reference(prepared: &PreparedDesign<'_>, slopes: bool) {
    let n = prepared.design.n_obs;
    let e = Elimination::new(prepared);
    let (reference, fast) = (ProjectedDesign::new(&e), build(&e));
    assert_eq!(e.eliminated_loadings().iter().any(Option::is_some), slopes);
    let order = &e.rows().order;
    assert!(LevelTasks::build(&e.rows().starts).0.len() > 1);
    let x: Vec<f64> = (0..fast.ncols()).map(|i| (i as f64).sin()).collect();
    let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
    let (mut expected, mut got) = (vec![0.0; n], vec![0.0; n]);
    reference.apply(&x, &mut expected).unwrap();
    fast.apply(&x, &mut got).unwrap();
    for (p, i) in order.rows(0..n).enumerate() {
        assert!((got[p] - expected[i as usize]).abs() < 1e-11);
    }
    let grouped: Vec<f64> = order.rows(0..n).map(|i| u[i as usize]).collect();
    let (mut expected, mut got) = (vec![0.0; x.len()], vec![0.0; x.len()]);
    reference.apply_adjoint(&u, &mut expected).unwrap();
    fast.apply_adjoint(&grouped, &mut got).unwrap();
    for (a, b) in got.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()));
    }
    let diagonal = fast.gram_diagonal();
    let mut column = vec![0.0; n];
    for (j, &d) in diagonal.iter().enumerate() {
        let unit: Vec<f64> = (0..x.len()).map(|i| (i == j) as u8 as f64).collect();
        reference.apply(&unit, &mut column).unwrap();
        let squared: f64 = column.iter().map(|v| v * v).sum();
        assert!(
            (d - squared).abs() < 1e-9 * (1.0 + squared),
            "column {j}: {d} vs {squared}"
        );
    }
    let mut projected = u.clone();
    project(&e, &mut projected, &mut vec![0.0; e.removed().len()]);
    let centered = fast.projected_rhs(&u);
    for (p, i) in order.rows(0..n).enumerate() {
        assert!((centered[p] - projected[i as usize]).abs() < 1e-11);
    }
}

#[test]
fn grouped_rows_reproduce_reference_with_a_large_level() {
    let n = 70_000;
    let categories = Array2::from_shape_fn((n, 3), |(i, j)| {
        [i.saturating_sub(50_000), i % 23, i % 7][j] as u32
    });
    let weights = weights(n);
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for weights in [None, Some(&weights[..])] {
                    let prepared =
                        PreparedDesign::new(categories.view().into_design().unwrap(), weights)
                            .unwrap();
                    assert_reproduces_reference(&prepared, false);
                }
            });
    }
}

#[test]
fn grouped_rows_reproduce_reference_across_channel_counts() {
    let n = 30_000;
    let weights = weights(n);
    for channels in [0, 1, 3, 4] {
        let categories = Array2::from_shape_fn((n, channels + 1), |(i, j)| {
            let row = i * 7919 % n;
            ((row * (2 * j + 1) + row / 7) % [5003, 17, 11, 7, 5][j]) as u32
        });
        let columns: Vec<Vec<u32>> = categories
            .columns()
            .into_iter()
            .map(|c| c.to_vec())
            .collect();
        let effects: Vec<_> = columns
            .iter()
            .map(|c| Effect::new(c, true, []).unwrap())
            .collect();
        for design in [
            Design::new(effects.clone()).unwrap(),
            Design::new_unsorted(effects).unwrap(),
        ] {
            for weights in [None, Some(&weights[..])] {
                let prepared =
                    PreparedDesign::new(design.clone().into_design().unwrap(), weights).unwrap();
                assert_reproduces_reference(&prepared, false);
            }
        }
    }
}

#[test]
fn grouped_rows_reproduce_the_projected_design() {
    let n = 30_000;
    let categories = Array2::from_shape_fn((n, 3), |(i, j)| {
        ((i * (2 * j + 1) + i / 7) % [5003, 131, 7][j]) as u32
    });
    let weights = weights(n);
    for weights in [None, Some(&weights[..])] {
        let design = Design::from_categories(categories.view()).unwrap();
        let prepared = PreparedDesign::new(design.into_design().unwrap(), weights).unwrap();
        assert_reproduces_reference(&prepared, false);
    }
}

#[test]
fn grouped_rows_reproduce_the_projected_slope_design() {
    let n = 30_000;
    let worker: Vec<u32> = (0..n).map(|i| (i % 3001) as u32).collect();
    let firm: Vec<u32> = (0..n).map(|i| ((i * 7 + i / 11) % 131) as u32).collect();
    let year: Vec<u32> = (0..n).map(|i| (i / 3001 % 10) as u32).collect();
    let z1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let z2: Vec<f64> = (0..n)
        .map(|i| year[i] as f64 + 1e-3 * (i as f64).cos())
        .collect();
    let z3: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos() + 2.0).collect();
    let weights = weights(n);
    for intercept in [true, false] {
        for weights in [None, Some(&weights[..])] {
            let effects = vec![
                Effect::new(&worker, intercept, [&z1[..], &z2[..]]).unwrap(),
                Effect::new(&firm, true, [&z3[..]]).unwrap(),
                Effect::new(&year, true, []).unwrap(),
            ];
            for design in [
                Design::new(effects.clone()).unwrap(),
                Design::new_unsorted(effects).unwrap(),
            ] {
                let prepared = PreparedDesign::new(design.into_design().unwrap(), weights).unwrap();
                assert_reproduces_reference(&prepared, true);
            }
        }
        // Kept slopes alone: the eliminated term is intercept-only.
        let effects = vec![
            Effect::new(&worker, true, []).unwrap(),
            Effect::new(&firm, intercept, [&z3[..]]).unwrap(),
            Effect::new(&year, true, [&z1[..]]).unwrap(),
        ];
        for design in [
            Design::new(effects.clone()).unwrap(),
            Design::new_unsorted(effects).unwrap(),
        ] {
            let prepared = PreparedDesign::new(design.into_design().unwrap(), None).unwrap();
            assert_reproduces_reference(&prepared, false);
        }
    }
}

/// Checks the projection against the full design operator, which shares none of its indexing.
fn assert_schur_gram(prepared: &PreparedDesign<'_>) {
    let n = prepared.design.n_obs;
    let e = Elimination::new(prepared);
    let op = ProjectedDesign::new(&e);
    let x: Vec<f64> = (0..op.ncols()).map(|i| (i as f64).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
    let mut ax = vec![0.0; n];
    let mut aty = vec![0.0; op.ncols()];
    op.apply(&x, &mut ax).unwrap();
    op.apply_adjoint(&y, &mut aty).unwrap();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    assert!((dot(&ax, &y) - dot(&x, &aty)).abs() < 1e-10 * (1.0 + dot(&ax, &y).abs()));
    let full_op = crate::operator::DesignOperator::new(prepared);
    let mut full = vec![0.0; prepared.design.n_dofs];
    full_op.apply_adjoint(&ax, &mut full).unwrap();
    let norm = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm(&full[e.removed()]) < 1e-10 * norm(&ax));
    let mut embedded = vec![0.0; prepared.design.n_dofs];
    e.embed(&x, &mut embedded);
    let mut gathered = vec![0.0; n];
    full_op.apply(&embedded, &mut gathered).unwrap();
    let mut coefficients = vec![0.0; e.removed().len()];
    fit_coefficients(&e, &gathered, &mut coefficients);
    for (target, b) in embedded[e.removed()].iter_mut().zip(coefficients) {
        *target = -b;
    }
    full_op.apply(&embedded, &mut gathered).unwrap();
    let mut gram = vec![0.0; prepared.design.n_dofs];
    full_op.apply_adjoint(&gathered, &mut gram).unwrap();
    let mut schur = vec![0.0; op.ncols()];
    e.restrict(&gram, &mut schur);
    op.apply_adjoint(&ax, &mut aty).unwrap();
    for (a, b) in aty.iter().zip(schur) {
        assert!((a - b).abs() < 1e-10 * (1.0 + b.abs()));
    }
}

#[test]
fn projected_operator_has_adjoint_and_schur_gram() {
    let categories = Array2::from_shape_fn((120, 3), |(i, j)| {
        ((i * (2 * j + 1) + i / 7) % [5, 13, 3][j]) as u32
    });
    let weights = weights(120);
    let design = Design::from_categories(categories.view()).unwrap();
    let prepared = PreparedDesign::new(design.into_design().unwrap(), Some(&weights)).unwrap();
    assert_schur_gram(&prepared);
}

#[test]
fn projected_slope_operator_has_adjoint_and_schur_gram() {
    let n = 600;
    let a: Vec<u32> = (0..n).map(|i| (i % 37) as u32).collect();
    let b: Vec<u32> = (0..n).map(|i| ((i * 7 + i / 11) % 13) as u32).collect();
    let c: Vec<u32> = (0..n).map(|i| (i / 37 % 5) as u32).collect();
    let z1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    // Within each `a` level, `z2` is `c` up to `1e-7`: a near alias the block inverse must survive.
    let z2: Vec<f64> = (0..n)
        .map(|i| c[i] as f64 + 1e-7 * (i as f64).cos())
        .collect();
    let z3: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos() + 2.0).collect();
    let weights = weights(n);
    for intercept in [true, false] {
        for weights in [None, Some(&weights[..])] {
            let effects = vec![
                Effect::new(&a, intercept, [&z1[..], &z2[..]]).unwrap(),
                Effect::new(&b, true, [&z3[..]]).unwrap(),
                Effect::new(&c, true, []).unwrap(),
            ];
            let prepared = PreparedDesign::new(effects.into_design().unwrap(), weights).unwrap();
            assert_schur_gram(&prepared);
        }
    }
}
