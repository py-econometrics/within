//! Two-phase deflated CG: run preconditioned CG to collect Krylov information,
//! extract Ritz pairs corresponding to slow modes, and return them for
//! downstream spectral analysis or restart heuristics.
//!
//! Phase 1: Run `pilot_iters` of preconditioned CG, recording the Lanczos basis.
//! Phase 2: Build tridiagonal from CG alpha/beta, eigendecompose, extract Ritz vectors.
//!
//! The caller can use these Ritz vectors and restart CG from the pilot solution.

use super::tridiagonal_eigen::tridiagonal_eigen;
use super::util::{dot, vec_norm, ColumnBasis};
use crate::Operator;

/// A Ritz pair extracted from the CG Krylov subspace.
pub struct RitzPair {
    /// Ritz value (approximate eigenvalue of the preconditioned operator M^{-1}A).
    pub ritz_value: f64,
    /// Ritz vector in the original DOF space (dense, length n_dofs).
    pub vector: Vec<f64>,
}

/// Result of the pilot CG phase.
pub struct PilotCgResult {
    /// Current solution after pilot iterations.
    pub x: Vec<f64>,
    /// Current residual after pilot iterations.
    pub residual: Vec<f64>,
    /// Residual norm.
    pub residual_norm: f64,
    /// Number of pilot iterations actually performed.
    pub iterations: usize,
    /// Whether CG converged during pilot phase.
    pub converged: bool,
    /// Extracted Ritz pairs, sorted by ascending Ritz value (smallest = slowest modes).
    pub ritz_pairs: Vec<RitzPair>,
}

/// Type alias for the Lanczos basis used in pilot CG.
type LanczosBasis = ColumnBasis;

/// Run a pilot phase of preconditioned CG and extract Ritz pairs.
///
/// Performs `pilot_iters` iterations of PCG on `A x = b` with preconditioner M,
/// recording the Krylov basis (scaled preconditioned residuals) and CG coefficients.
/// Then builds the Lanczos tridiagonal, eigendecomposes it, and returns the
/// `n_ritz` Ritz pairs with smallest Ritz values (the slow modes).
///
/// The returned `PilotCgResult` includes the current solution and residual,
/// allowing the caller to restart CG with an improved preconditioner.
pub fn pilot_cg_with_ritz<A: Operator, M: Operator>(
    operator: &A,
    preconditioner: &M,
    b: &[f64],
    tol: f64,
    pilot_iters: usize,
    n_ritz: usize,
) -> PilotCgResult {
    let n = operator.ncols();
    debug_assert_eq!(b.len(), n);

    let b_norm = vec_norm(b);
    if b_norm == 0.0 || pilot_iters == 0 {
        return PilotCgResult {
            x: vec![0.0; n],
            residual: b.to_vec(),
            residual_norm: b_norm,
            iterations: 0,
            converged: b_norm == 0.0,
            ritz_pairs: vec![],
        };
    }

    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    let mut ap = vec![0.0; n];

    preconditioner.apply(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    // Record Lanczos basis vectors: v_k = z_k / sqrt(rz_k)
    // and CG coefficients alpha_k, beta_k
    let mut lanczos_basis = LanczosBasis::new(pilot_iters, n);
    let mut alphas: Vec<f64> = Vec::with_capacity(pilot_iters);
    let mut betas: Vec<f64> = Vec::with_capacity(pilot_iters);

    // Record first Lanczos vector
    let scale = 1.0 / rz.sqrt();
    lanczos_basis.push_scaled(&z, scale);

    let mut r_norm = b_norm;
    let mut actual_iters = 0;

    for itn in 0..pilot_iters {
        operator.apply(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            actual_iters = itn + 1;
            break;
        }
        let alpha = rz / pap;
        alphas.push(alpha);

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        r_norm = vec_norm(&r);
        actual_iters = itn + 1;

        if r_norm / b_norm <= tol {
            break;
        }

        if itn + 1 >= pilot_iters {
            break;
        }

        preconditioner.apply(&r, &mut z);
        let rz_new = dot(&r, &z);
        if rz_new.abs() < 1e-300 {
            break;
        }
        let beta = rz_new / rz;
        betas.push(beta);

        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        // Record Lanczos vector
        let scale = 1.0 / rz_new.sqrt();
        lanczos_basis.push_scaled(&z, scale);

        rz = rz_new;
    }

    let converged = r_norm / b_norm <= tol;

    // Build tridiagonal matrix from CG coefficients
    let m = alphas.len(); // number of completed iterations
    let ritz_pairs = if m >= 2 && n_ritz > 0 {
        extract_ritz_pairs(&alphas, &betas, &lanczos_basis, n_ritz)
    } else {
        vec![]
    };

    PilotCgResult {
        x,
        residual: r,
        residual_norm: r_norm,
        iterations: actual_iters,
        converged,
        ritz_pairs,
    }
}

/// Build tridiagonal from CG alpha/beta and extract Ritz pairs.
fn extract_ritz_pairs(
    alphas: &[f64],
    betas: &[f64],
    lanczos_basis: &LanczosBasis,
    n_ritz: usize,
) -> Vec<RitzPair> {
    let m = alphas.len();
    if m < 2 {
        return vec![];
    }

    // Build tridiagonal T from CG recurrence.
    // T[k,k] = 1/alpha_k + beta_{k-1}/alpha_{k-1}  (k > 0)
    // T[0,0] = 1/alpha_0
    // T[k,k-1] = T[k-1,k] = sqrt(beta_{k-1}) / alpha_{k-1}
    let mut diag = vec![0.0; m];
    let mut off_diag = vec![0.0; m - 1];

    diag[0] = 1.0 / alphas[0];
    for k in 1..m {
        let beta_prev = if k - 1 < betas.len() {
            betas[k - 1]
        } else {
            0.0
        };
        diag[k] = 1.0 / alphas[k] + beta_prev / alphas[k - 1];
        off_diag[k - 1] = beta_prev.sqrt() / alphas[k - 1];
    }

    // Eigendecompose the tridiagonal
    let (eigenvalues, eigenvectors) = tridiagonal_eigen(&diag, &off_diag);

    // Sort by ascending eigenvalue
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

    // Take the n_ritz smallest (positive) eigenvalues
    let n_select = n_ritz.min(m).min(lanczos_basis.len());
    let n_dofs = lanczos_basis.n;

    let mut pairs = Vec::with_capacity(n_select);
    for &idx in indices.iter().take(n_select) {
        let lambda = eigenvalues[idx];
        if lambda <= 0.0 {
            continue;
        }

        // Ritz vector = Σ_k s[k] * v_k where s is the eigenvector of T
        // and v_k is the k-th Lanczos basis vector
        let mut y = vec![0.0; n_dofs];
        let n_basis = lanczos_basis.len().min(m);
        for k in 0..n_basis {
            let s_k = eigenvectors[k * m + idx];
            if s_k.abs() > 1e-15 {
                for (yi, &basis_val) in y.iter_mut().zip(lanczos_basis.col(k)) {
                    *yi += s_k * basis_val;
                }
            }
        }

        // Normalize
        let norm = vec_norm(&y);
        if norm > 1e-14 {
            for yi in &mut y {
                *yi /= norm;
            }
            pairs.push(RitzPair {
                ritz_value: lambda,
                vector: y,
            });
        }
    }

    pairs
}

/// Convert dense Ritz vectors to sparse basis vectors.
///
/// Drops entries with absolute value below `drop_tol`.
pub fn ritz_pairs_to_sparse_basis(pairs: &[RitzPair], drop_tol: f64) -> Vec<(Vec<u32>, Vec<f64>)> {
    pairs
        .iter()
        .map(|pair| {
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for (i, &v) in pair.vector.iter().enumerate() {
                if v.abs() > drop_tol {
                    indices.push(i as u32);
                    values.push(v);
                }
            }
            (indices, values)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ritz_pairs_simple() {
        // 3 CG iterations with known coefficients
        let alphas = vec![0.5, 0.3, 0.2];
        let betas = vec![0.4, 0.3];
        let n = 10;

        // Create simple Lanczos basis (orthonormal unit vectors)
        let mut basis = LanczosBasis::new(3, n);
        for k in 0..3 {
            let mut v = vec![0.0; n];
            v[k] = 1.0;
            basis.push_scaled(&v, 1.0);
        }

        let pairs = extract_ritz_pairs(&alphas, &betas, &basis, 2);
        assert!(!pairs.is_empty());
        // Ritz values should be positive
        for p in &pairs {
            assert!(p.ritz_value > 0.0);
            assert_eq!(p.vector.len(), n);
        }
        // Should be sorted ascending
        for w in pairs.windows(2) {
            assert!(w[0].ritz_value <= w[1].ritz_value);
        }
    }
}
