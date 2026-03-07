use super::vec_norm;
use crate::{Operator, SolveError};

/// Inner product of two vectors.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Column basis storage
// ---------------------------------------------------------------------------

/// Column-major flat storage for basis vectors.
///
/// Stores up to `capacity` vectors of dimension `n` in a single contiguous
/// allocation. Avoids per-restart/per-iteration heap allocation.
struct ColumnBasis {
    data: Vec<f64>,
    /// Dimension of each vector.
    n: usize,
    len: usize,
}

impl ColumnBasis {
    /// Create with room for `capacity` vectors of length `n`.
    fn new(capacity: usize, n: usize) -> Self {
        Self {
            data: vec![0.0; capacity * n],
            n,
            len: 0,
        }
    }

    /// Return column `j` as a shared slice.
    #[inline]
    fn col(&self, j: usize) -> &[f64] {
        debug_assert!(j < self.len);
        let start = j * self.n;
        &self.data[start..start + self.n]
    }

    /// Return column `j` as a mutable slice.
    ///
    /// # Safety
    /// Caller must ensure no aliasing mutable references to the same column.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn col_mut(&self, j: usize) -> &mut [f64] {
        debug_assert!(j < self.len);
        let start = j * self.n;
        unsafe {
            let ptr = self.data.as_ptr().add(start) as *mut f64;
            std::slice::from_raw_parts_mut(ptr, self.n)
        }
    }

    /// Append `src` as the next column, incrementing `len`.
    fn push_from(&mut self, src: &[f64]) {
        debug_assert_eq!(src.len(), self.n);
        let start = self.len * self.n;
        self.data[start..start + self.n].copy_from_slice(src);
        self.len += 1;
    }

    /// Append a zero-filled column, incrementing `len`.
    fn push_zeroed(&mut self) {
        let start = self.len * self.n;
        self.data[start..start + self.n].fill(0.0);
        self.len += 1;
    }

    /// Reset the number of stored vectors to zero (does not deallocate).
    fn clear(&mut self) {
        self.len = 0;
    }
}

// ---------------------------------------------------------------------------
// GMRES result
// ---------------------------------------------------------------------------

/// Result of a GMRES solve.
#[must_use]
pub struct GmresResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Whether GMRES converged within the tolerance.
    pub converged: bool,
    /// Total number of iterations performed (across all restarts).
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
}

// ---------------------------------------------------------------------------
// Flat storage helpers
// ---------------------------------------------------------------------------

/// Flat column-major storage for the upper Hessenberg matrix.
///
/// Column `j` has `j + 2` active entries (rows `0..=j+1`).
/// Storage is allocated for `max_cols` columns, each with `max_cols + 1` rows.
struct HessenbergMatrix {
    data: Vec<f64>,
    max_cols: usize,
    stride: usize,
}

impl HessenbergMatrix {
    fn new(max_cols: usize) -> Self {
        let stride = max_cols + 1;
        Self {
            data: vec![0.0; max_cols * stride],
            max_cols,
            stride,
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(col < self.max_cols && row < self.stride);
        self.data[col * self.stride + row]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, val: f64) {
        debug_assert!(col < self.max_cols && row < self.stride);
        self.data[col * self.stride + row] = val;
    }

    fn clear(&mut self) {
        self.data.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Arnoldi cycle
// ---------------------------------------------------------------------------

/// Mutable scratch buffers used by the Arnoldi cycle.
struct ArnoldiState<'a> {
    v_basis: &'a mut ColumnBasis,
    z_basis: &'a mut ColumnBasis,
    h: &'a mut HessenbergMatrix,
    cs: &'a mut [f64],
    sn: &'a mut [f64],
    g: &'a mut [f64],
    w: &'a mut [f64],
}

enum ArnoldiOutcome {
    Converged { residual: f64 },
    MaxIter { residual: f64 },
    Breakdown,
    RestartNeeded,
}

/// Run one Arnoldi cycle (inner GMRES loop).
///
/// Returns `(outcome, j)` where `j` is the number of Arnoldi steps completed
/// in this cycle.  The caller is responsible for solving the upper-triangular
/// system and updating `x` once, regardless of the outcome.
fn arnoldi_cycle<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
    state: &mut ArnoldiState,
    abs_tol: f64,
    iters_this_cycle: usize,
    total_iters: &mut usize,
    maxiter: usize,
) -> Result<(ArnoldiOutcome, usize), SolveError> {
    let mut j = 0;

    while j < iters_this_cycle {
        // z_j = M^{-1} v_j
        {
            let v_j = state.v_basis.col(j);
            state.z_basis.push_zeroed();
            let z_j = state.z_basis.col_mut(j);
            preconditioner.try_apply(v_j, z_j)?;
        }

        // w = A z_j
        {
            let z_j = state.z_basis.col(j);
            operator.try_apply(z_j, state.w)?;
        }

        // Modified Gram-Schmidt orthogonalisation
        for i in 0..=j {
            let hij = dot(state.w, state.v_basis.col(i));
            state.h.set(i, j, hij);
            let v_i = state.v_basis.col(i);
            for (wk, &vi) in state.w.iter_mut().zip(v_i) {
                *wk -= hij * vi;
            }
        }
        let h_jp1_j = vec_norm(state.w);
        state.h.set(j + 1, j, h_jp1_j);

        // Apply previous Givens rotations to the new column
        for i in 0..j {
            let h_ij = state.h.get(i, j);
            let h_i1j = state.h.get(i + 1, j);
            let temp = state.cs[i] * h_ij + state.sn[i] * h_i1j;
            state
                .h
                .set(i + 1, j, -state.sn[i] * h_ij + state.cs[i] * h_i1j);
            state.h.set(i, j, temp);
        }

        // Compute new Givens rotation for row (j, j+1)
        let (c, s) = givens_rotation(state.h.get(j, j), state.h.get(j + 1, j));
        state.cs[j] = c;
        state.sn[j] = s;

        // Apply to h column j
        let h_jj = state.h.get(j, j);
        let h_j1j = state.h.get(j + 1, j);
        state.h.set(j, j, c * h_jj + s * h_j1j);
        state.h.set(j + 1, j, 0.0);

        // Apply to g
        let temp = c * state.g[j] + s * state.g[j + 1];
        state.g[j + 1] = -s * state.g[j] + c * state.g[j + 1];
        state.g[j] = temp;

        j += 1;
        *total_iters += 1;

        let residual = state.g[j].abs();

        if residual <= abs_tol {
            return Ok((ArnoldiOutcome::Converged { residual }, j));
        }

        if *total_iters >= maxiter {
            return Ok((ArnoldiOutcome::MaxIter { residual }, j));
        }

        // Extend basis if not at last iteration
        if h_jp1_j > 1e-300 {
            // Normalise w into next basis vector
            let inv = 1.0 / h_jp1_j;
            for val in state.w.iter_mut() {
                *val *= inv;
            }
            state.v_basis.push_from(state.w);
        } else {
            // Lucky breakdown: Krylov subspace exhausted
            return Ok((ArnoldiOutcome::Breakdown, j));
        }
    }

    Ok((ArnoldiOutcome::RestartNeeded, j))
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Right-preconditioned GMRES(m) with restarts.
///
/// Solves A x = b using right preconditioning: A M^{-1} (M x) = b.
/// Uses Arnoldi iteration with Modified Gram-Schmidt orthogonalisation
/// and Givens rotations to solve the Hessenberg least-squares problem.
///
/// `restart`: Krylov subspace dimension before restart (m in GMRES(m)).
pub fn gmres_solve<A: Operator + ?Sized, M: Operator + ?Sized>(
    operator: &A,
    preconditioner: &M,
    b: &[f64],
    tol: f64,
    maxiter: usize,
    restart: usize,
) -> Result<GmresResult, SolveError> {
    let n = operator.ncols();
    debug_assert_eq!(b.len(), n);

    let b_norm = vec_norm(b);
    if b_norm == 0.0 {
        return Ok(GmresResult {
            x: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    let abs_tol = tol * b_norm;
    let m = restart;

    let mut x = vec![0.0; n];
    let mut total_iters = 0;

    // Working vectors reused across restarts
    let mut r = vec![0.0; n];
    let mut w = vec![0.0; n];

    // Pre-allocate Krylov storage once
    let mut v_basis = ColumnBasis::new(m + 1, n);
    let mut z_basis = ColumnBasis::new(m, n);
    let mut h = HessenbergMatrix::new(m);
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];

    loop {
        // r = b - A x
        operator.try_apply(&x, &mut r)?;
        for (ri, &bi) in r.iter_mut().zip(b) {
            *ri = bi - *ri;
        }
        let beta = vec_norm(&r);

        if beta <= abs_tol {
            return Ok(GmresResult {
                x,
                converged: true,
                iterations: total_iters,
                residual_norm: beta,
            });
        }

        if total_iters >= maxiter {
            return Ok(GmresResult {
                x,
                converged: false,
                iterations: total_iters,
                residual_norm: beta,
            });
        }

        // Reset storage for this restart cycle
        v_basis.clear();
        z_basis.clear();
        h.clear();
        cs.fill(0.0);
        sn.fill(0.0);
        g.fill(0.0);
        g[0] = beta;

        // v_0 = r / beta
        let inv_beta = 1.0 / beta;
        for val in r.iter_mut() {
            *val *= inv_beta;
        }
        v_basis.push_from(&r);

        let iters_this_cycle = (maxiter - total_iters).min(m);

        let (outcome, j) = arnoldi_cycle(
            operator,
            preconditioner,
            &mut ArnoldiState {
                v_basis: &mut v_basis,
                z_basis: &mut z_basis,
                h: &mut h,
                cs: &mut cs,
                sn: &mut sn,
                g: &mut g,
                w: &mut w,
            },
            abs_tol,
            iters_this_cycle,
            &mut total_iters,
            maxiter,
        )?;

        // Solve upper triangular system and update x -- done ONCE per cycle
        let y = solve_upper_triangular(&h, &g, j);
        update_solution(&mut x, &z_basis, &y, j);

        match outcome {
            ArnoldiOutcome::Converged { residual } => {
                return Ok(GmresResult {
                    x,
                    converged: true,
                    iterations: total_iters,
                    residual_norm: residual,
                });
            }
            ArnoldiOutcome::MaxIter { residual } => {
                return Ok(GmresResult {
                    x,
                    converged: false,
                    iterations: total_iters,
                    residual_norm: residual,
                });
            }
            ArnoldiOutcome::Breakdown => {
                // Lucky breakdown: recompute actual residual
                operator.try_apply(&x, &mut r)?;
                for (ri, &bi) in r.iter_mut().zip(b) {
                    *ri = bi - *ri;
                }
                let res = vec_norm(&r);
                return Ok(GmresResult {
                    x,
                    converged: res <= abs_tol,
                    iterations: total_iters,
                    residual_norm: res,
                });
            }
            ArnoldiOutcome::RestartNeeded => {
                // Loop continues with updated x
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute Givens rotation coefficients (c, s) such that
/// [c  s] [a]   [r]
/// [-s c] [b] = [0]
#[inline]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if a == 0.0 {
        (0.0, 1.0)
    } else if a.abs() > b.abs() {
        let t = b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, c * t)
    } else {
        let t = a / b;
        let s = 1.0 / (1.0 + t * t).sqrt();
        (s * t, s)
    }
}

/// Solve the upper triangular system R y = g[0..k]
/// where R is stored in the Hessenberg matrix (after Givens rotations).
#[inline]
fn solve_upper_triangular(h: &HessenbergMatrix, g: &[f64], k: usize) -> Vec<f64> {
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = g[i];
        for (j, yj) in y.iter().enumerate().take(k).skip(i + 1) {
            sum -= h.get(i, j) * yj;
        }
        y[i] = sum / h.get(i, i);
    }
    y
}

/// Update x += Z * y
#[inline]
fn update_solution(x: &mut [f64], z_basis: &ColumnBasis, y: &[f64], k: usize) {
    for (j, &yj) in y.iter().enumerate().take(k) {
        let z_j = z_basis.col(j);
        for (xi, &zi) in x.iter_mut().zip(z_j.iter()) {
            *xi += yj * zi;
        }
    }
}
