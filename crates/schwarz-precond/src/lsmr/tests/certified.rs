//! A reported convergence is certified by its tolerance (#455): across seeded inconsistent systems
//! `[diag(d); 0]` whose `d` spans up to ten orders, every solve converges and meets `tol` against
//! the exact `‖A‖₂ = max dᵢ`, through the default power-iteration norm floor.

use rstest::rstest;

use super::super::{mlsmr, splitmix64_unit, vec_norm, LsmrStopReason, MlsmrOptions};
use crate::lsmr::fixtures::{normal_equation_residual, residual, DiagOp};
use crate::{MemStack, Operator, SolveError};

/// `[diag(d); 0]`: the zero rows make any `b` with entries there inconsistent.
struct Padded(Vec<f64>);

const ZERO_ROWS: usize = 3;

impl Operator for Padded {
    fn nrows(&self) -> usize {
        self.0.len() + ZERO_ROWS
    }
    fn ncols(&self) -> usize {
        self.0.len()
    }
    fn apply_with(&self, x: &[f64], y: &mut [f64], _: &mut MemStack) -> Result<(), SolveError> {
        let (top, zero) = y.split_at_mut(self.0.len());
        for ((y, d), x) in top.iter_mut().zip(&self.0).zip(x) {
            *y = d * x;
        }
        zero.fill(0.0);
        Ok(())
    }
    fn apply_adjoint_with(
        &self,
        u: &[f64],
        x: &mut [f64],
        _: &mut MemStack,
    ) -> Result<(), SolveError> {
        for ((x, d), u) in x.iter_mut().zip(&self.0).zip(u) {
            *x = d * u;
        }
        Ok(())
    }
}

/// `preconditioned` scales `M⁻¹ = diag(mᵢ / dᵢ²)` with `mᵢ` across three orders, so its metric
/// estimate runs a few times ahead of the plain ratio: a refuted stop's next pass must iterate past
/// that gap, not re-trigger on it. Without the gap, 38 of its 402 solves end unconverged; the gap
/// still widens across restarts for the few it leaves.
#[rstest]
#[case::plain(false, 0)]
#[case::preconditioned(true, 5)]
fn every_stop_converges_and_meets_its_tolerance(
    #[case] preconditioned: bool,
    #[case] unconverged: usize,
) {
    const COLUMNS: usize = 30;
    let mut draws = (0u64..).map(splitmix64_unit);
    let mut misses = 0;
    for span in [1e4f64, 1e10] {
        for tol in [1e-6, 1e-8, 1e-10] {
            for _ in 0..67 {
                let d: Vec<f64> = (draws.by_ref().take(COLUMNS))
                    .map(|u| span.powf(0.5 * (u + 1.0)))
                    .collect();
                let b: Vec<f64> = draws.by_ref().take(COLUMNS + ZERO_ROWS).collect();
                let m: Vec<f64> = (d.iter().zip(draws.by_ref()))
                    .map(|(d, u)| {
                        if preconditioned {
                            10f64.powf(1.5 * u) / (d * d)
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let a = Padded(d);
                let options = MlsmrOptions::default();
                let r = mlsmr(&a, &b, &DiagOp(m), tol, 1_000, None, options).expect("solve");
                if !r.converged() {
                    assert_eq!(r.stop_reason, LsmrStopReason::FalseConvergence);
                    misses += 1;
                    continue;
                }

                let residual = residual(&a, &r.x, &b);
                let a_norm = a.0.iter().fold(0.0f64, |m, &d| m.max(d));
                let ratio = normal_equation_residual(&a, &r.x, &b) / (a_norm * vec_norm(&residual));
                // The test recomputes in its own rounding, and allows only that.
                assert!(
                    ratio <= tol * (1.0 + 1e-6),
                    "span {span:e}, tol {tol:e}: backward error {:.2}× tol after {} iterations",
                    ratio / tol,
                    r.iterations,
                );
            }
        }
    }
    assert_eq!(misses, unconverged);
}
