//! Small dense symmetric eigenvalue solver using Jacobi rotations.
//!
//! Designed for matrices up to ~50×50 (tridiagonal CG coefficient matrices).

/// Compute all eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// Given diagonal `d` (length n) and off-diagonal `e` (length n-1), computes
/// the eigendecomposition `T = V Λ V^T`.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is n×n row-major,
/// with `eigenvectors[j*n .. (j+1)*n]` being the j-th eigenvector.
pub fn tridiagonal_eigen(d: &[f64], e: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = d.len();
    assert_eq!(e.len(), n - 1);
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![d[0]], vec![1.0]);
    }

    // Build full symmetric matrix from tridiagonal
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = d[i];
    }
    for i in 0..n - 1 {
        a[i * n + (i + 1)] = e[i];
        a[(i + 1) * n + i] = e[i];
    }

    symmetric_eigen_inplace(&mut a, n)
}

/// Jacobi eigenvalue algorithm for a dense symmetric matrix (in-place).
///
/// `a` is n×n row-major. On return, `a` contains the diagonal eigenvalues.
/// Returns `(eigenvalues, eigenvectors)`.
fn symmetric_eigen_inplace(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Initialize eigenvector matrix to identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 100;
    let tol = 1e-14;

    for _ in 0..max_sweeps {
        // Find max off-diagonal element
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                max_off = max_off.max(a[i * n + j].abs());
            }
        }
        if max_off < tol {
            break;
        }

        // Sweep: rotate all off-diagonal elements
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < tol * 0.01 {
                    continue;
                }

                let diff = a[q * n + q] - a[p * n + p];
                let t = if diff.abs() < 1e-300 {
                    // a[p][p] ≈ a[q][q], rotation angle = π/4
                    1.0f64.copysign(apq)
                } else {
                    let tau = diff / (2.0 * apq);
                    // Smaller root of t^2 + 2τt - 1 = 0
                    let sign = 1.0f64.copysign(tau);
                    sign / (tau.abs() + (tau * tau + 1.0).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_rot = s / (1.0 + c);

                // Update matrix A
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                a[p * n + p] -= t * apq;
                a[q * n + q] += t * apq;

                // Rotate rows/cols p and q
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = arp - s * (arq + tau_rot * arp);
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = arq + s * (arp - tau_rot * arq);
                    a[q * n + r] = a[r * n + q];
                }

                // Accumulate eigenvectors
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = vrp - s * (vrq + tau_rot * vrp);
                    v[r * n + q] = vrq + s * (vrp - tau_rot * vrq);
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tridiagonal_2x2() {
        let (vals, vecs) = tridiagonal_eigen(&[2.0, 3.0], &[1.0]);
        let mut sorted: Vec<f64> = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Eigenvalues of [[2,1],[1,3]] are (5 ± sqrt(5))/2
        let expected_min = (5.0 - 5.0f64.sqrt()) / 2.0;
        let expected_max = (5.0 + 5.0f64.sqrt()) / 2.0;
        assert!((sorted[0] - expected_min).abs() < 1e-12);
        assert!((sorted[1] - expected_max).abs() < 1e-12);

        // Check orthogonality
        let n = 2;
        let dot: f64 = (0..n).map(|i| vecs[i * n] * vecs[i * n + 1]).sum();
        assert!(dot.abs() < 1e-12);
    }

    #[test]
    fn test_tridiagonal_identity() {
        let (vals, vecs) = tridiagonal_eigen(&[1.0, 1.0, 1.0], &[0.0, 0.0]);
        for &v in &vals {
            assert!((v - 1.0).abs() < 1e-12);
        }
        // Eigenvectors should be orthogonal
        let n = 3;
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f64 = (0..n).map(|k| vecs[k * n + i] * vecs[k * n + j]).sum();
                assert!(dot.abs() < 1e-10, "i={i} j={j} dot={dot}");
            }
        }
    }

    #[test]
    fn test_tridiagonal_4x4() {
        // T = tridiag(1, 2, 1) of size 4
        let d = vec![2.0; 4];
        let e = vec![1.0; 3];
        let (vals, vecs) = tridiagonal_eigen(&d, &e);
        let n = 4;

        // Verify T v = λ v for each eigenpair
        for j in 0..n {
            let lambda = vals[j];
            let v: Vec<f64> = (0..n).map(|i| vecs[i * n + j]).collect();
            for i in 0..n {
                let mut tv_i = d[i] * v[i];
                if i > 0 {
                    tv_i += e[i - 1] * v[i - 1];
                }
                if i < n - 1 {
                    tv_i += e[i] * v[i + 1];
                }
                assert!(
                    (tv_i - lambda * v[i]).abs() < 1e-10,
                    "j={j} i={i} Tv={tv_i} lv={}",
                    lambda * v[i]
                );
            }
        }
    }
}
