//! LSMR windowed reorthogonalization on an ill-conditioned problem.
//!
//! Demonstrates the regime where windowed modified Gram-Schmidt actually
//! pays off: a Vandermonde-style least-squares problem with `cond(A) ~ 1e9`,
//! where the plain short recurrence loses `v`-orthogonality and convergence
//! stalls. Doubling the polynomial degree raises the condition number further
//! and makes the gap between `local_size = 0` and `local_size > 0` dramatic.
//!
//! Run:
//!   cargo run --release --example lsmr_local_reorth -p schwarz-precond

use std::time::Instant;

use schwarz_precond::solve::lsmr::mlsmr;
use schwarz_precond::{IdentityOperator, Operator};

/// Dense row-major operator. Used here to build a Vandermonde-flavored
/// least-squares problem `A[i,j] = (i / (rows-1))^j`.
struct DenseOp {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseOp {
    fn vandermonde(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            let x = i as f64 / (rows - 1).max(1) as f64;
            let mut p = 1.0;
            for j in 0..cols {
                data[i * cols + j] = p;
                p *= x;
            }
        }
        Self { rows, cols, data }
    }
}

impl Operator for DenseOp {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for (yi, row) in y.iter_mut().zip(self.data.chunks_exact(self.cols)) {
            *yi = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
    fn apply_adjoint(&self, u: &[f64], x: &mut [f64]) {
        for (j, xj) in x.iter_mut().enumerate() {
            let mut s = 0.0;
            for (ui, row) in u.iter().zip(self.data.chunks_exact(self.cols)) {
                s += row[j] * ui;
            }
            *xj = s;
        }
    }
}

fn run(label: &str, rows: usize, cols: usize, tol: f64, maxiter: usize) {
    let op = DenseOp::vandermonde(rows, cols);
    // Smooth target: log(1 + x) sampled on the grid. Polynomial-approximable,
    // so the least-squares residual is near zero — making the iteration count
    // depend on conditioning, not on the residual floor.
    let b: Vec<f64> = (0..rows)
        .map(|i| {
            let x = i as f64 / (rows - 1) as f64;
            (1.0 + x).ln()
        })
        .collect();

    println!("\n=== {label} (rows={rows}, cols={cols}, tol={tol:.0e}) ===");
    println!(
        "{:>5}  {:>7}  {:>10}  {:>12}  {:>5}",
        "local", "iters", "time_ms", "‖A^T r‖", "conv"
    );
    println!("{}", "-".repeat(48));

    for &local_size in &[None, Some(5), Some(10), Some(20), Some(40)] {
        if local_size.unwrap_or(0) > cols {
            continue;
        }
        let t0 = Instant::now();
        let r = mlsmr(&op, &b, None::<&IdentityOperator>, tol, maxiter, local_size).expect("solve");
        let dt = t0.elapsed().as_secs_f64() * 1e3;

        // Compute the normal-equation residual ‖Aᵀ(b - Ax)‖ as a proxy for
        // solution quality, since LSMR's internal estimate ignores drift.
        let mut ax = vec![0.0; rows];
        op.apply(&r.x, &mut ax);
        let resid: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
        let mut atr = vec![0.0; cols];
        op.apply_adjoint(&resid, &mut atr);
        let normar = atr.iter().map(|x| x * x).sum::<f64>().sqrt();

        let local_label = match local_size {
            None => "off".to_string(),
            Some(n) => n.to_string(),
        };
        println!(
            "{:>5}  {:>7}  {:>10.2}  {:>12.2e}  {:>5}",
            local_label,
            r.iterations,
            dt,
            normar,
            if r.converged { "yes" } else { "NO" }
        );
    }
}

fn main() {
    // Mild ill-conditioning: cols=12 → cond ~ 1e10. Plain LSMR stalls past
    // ~12 iterations because the v's lose Euclidean orthogonality.
    run("Vandermonde mild", 60, 12, 1e-10, 1000);

    // Stronger: cols=18 → cond ~ 1e15. Plain LSMR cannot reach 1e-10 at all.
    run("Vandermonde harsh", 80, 18, 1e-10, 1000);

    // Tight tolerance amplifies the gap further.
    run("Vandermonde tight tol", 60, 14, 1e-13, 1000);
}
