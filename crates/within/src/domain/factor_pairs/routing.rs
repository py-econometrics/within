//! Balance + scale routing for signed components.
//!
//! A signed component's Gram `[D_q, C; Cᵀ, D_r]` carries arbitrary-sign cross
//! cells that the SDDM-only local solver cannot take as-is.
//! [`balance_and_scale`] produces the congruence `d = σ ⊙ λ`: a ±1 signature
//! `σ` folding every cell non-negative (none exists on a negative-sign cycle
//! — [`Frustrated`]) times a positive diagonal `λ` improving weak diagonal
//! dominance. σ is exact; λ is quality-only (approx-chol clamps residual
//! deficits), so the congruence-transformed solve is exact for any λ > 0.

use crate::domain::{BlockDiagonals, CrossTab};

/// Acceptance slack for weak diagonal dominance: rows within
/// `diag·(1 + WDD_SLACK)` of their absolute off-diagonal sum count as
/// dominant. Residual deficits ≤ this are clamped by approx-chol
/// (quality-only); wide enough that exactly-singular boundary components exit
/// the relaxation in a few sweeps instead of chasing rounding.
const WDD_SLACK: f64 = 1e-6;

/// Relaxation budget; exhaustion hands the current λ over as-is.
const MAX_SWEEPS: usize = 64;

/// A negative-sign cycle: no ±1 signature folds every cross cell non-negative.
#[derive(Debug)]
pub(super) struct Frustrated;

/// Compute the congruence `d = σ ⊙ λ` over the `[q | r]` local DOF layout of
/// one connected multi-node component; `Ok(None)` when `d ≡ 1` suffices
/// (already non-negative and weakly diagonally dominant).
pub(super) fn balance_and_scale(
    cross_tab: &CrossTab,
    diagonals: &BlockDiagonals,
) -> Result<Option<Box<[f64]>>, Frustrated> {
    let n_q = cross_tab.n_q();
    let n = cross_tab.n_local();
    // Connected component: every node has an edge, and any node with an edge
    // has a strictly positive diagonal (each nonzero cell contributes w·l² > 0
    // to both endpoint diagonals), so 1/√D below is safe.
    let diag = |i: usize| {
        if i < n_q {
            diagonals.q[i]
        } else {
            diagonals.r[i - n_q]
        }
    };
    // Row `i` of the symmetric cross structure: q-nodes walk C, r-nodes walk
    // Cᵀ; neighbor indices come back local to the opposite block.
    let row = |i: usize| {
        let (block, r, off) = if i < n_q {
            (&cross_tab.c, i, n_q)
        } else {
            (&cross_tab.ct, i - n_q, 0)
        };
        let lo = block.indptr[r] as usize;
        let hi = block.indptr[r + 1] as usize;
        block.indices[lo..hi]
            .iter()
            .zip(&block.data[lo..hi])
            .map(move |(&j, &v)| (off + j as usize, v))
    };

    // σ: sign the spanning tree of a DFS from node 0 so every tree edge folds
    // positive (0.0 marks unvisited; cells are nonzero, so signum is ±1).
    let mut sigma = vec![0.0f64; n];
    sigma[0] = 1.0;
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        for (j, v) in row(i) {
            if sigma[j] == 0.0 {
                sigma[j] = sigma[i] * v.signum();
                stack.push(j);
            }
        }
    }

    // Frustration: a non-tree edge violating the signature is a negative
    // cycle; ±1 folding is exact, so acceptance here means the consumer's
    // `v ≥ 0` assertion holds exactly.
    for i in 0..n_q {
        for (j, v) in row(i) {
            if sigma[i] * sigma[j] * v < 0.0 {
                return Err(Frustrated);
            }
        }
    }

    // Already weakly diagonally dominant (σ-independent): λ ≡ 1.
    let row_abs: Vec<f64> = (0..n).map(|i| row(i).map(|(_, v)| v.abs()).sum()).collect();
    if (0..n).all(|i| diag(i) * (1.0 + WDD_SLACK) >= row_abs[i]) {
        return Ok(if sigma.iter().all(|&s| s == 1.0) {
            None
        } else {
            Some(sigma.into())
        });
    }

    // λ in Jacobi-normalized coordinates `λ_i = μ_i/√D_i`: WDD becomes
    // `μ_i ≥ Σ_j n_ij μ_j` with `n_ij = |c_ij|/√(D_i D_j)`. Monotone block
    // Gauss–Seidel (q-side then r-side, deterministic order — defeats the
    // period-2 stall of bipartite power iteration) raises deficient rows; a
    // no-op sweep certifies every row within slack of the final μ.
    let inv_sqrt_d: Vec<f64> = (0..n).map(|i| 1.0 / diag(i).sqrt()).collect();
    let mut mu = vec![1.0f64; n];
    for _ in 0..MAX_SWEEPS {
        let mut raised = false;
        for i in 0..n {
            let t: f64 = row(i)
                .map(|(j, v)| v.abs() * inv_sqrt_d[i] * inv_sqrt_d[j] * mu[j])
                .sum();
            if t > mu[i] * (1.0 + WDD_SLACK) {
                mu[i] = t;
                raised = true;
            }
        }
        if !raised {
            break;
        }
    }
    Ok(Some(
        (0..n).map(|i| sigma[i] * mu[i] * inv_sqrt_d[i]).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_block::CsrBlock;

    fn cross_tab_of(table: &[f64], n_q: usize, n_r: usize) -> CrossTab {
        let c = CsrBlock::from_dense_table(table, n_q, n_r);
        let ct = c.transpose();
        CrossTab { c, ct }
    }

    /// `d·A·d` is weakly diagonally dominant within slack on every row.
    fn assert_wdd(cross_tab: &CrossTab, diagonals: &BlockDiagonals, d: &[f64]) {
        let n_q = cross_tab.n_q();
        for i in 0..n_q {
            let lo = cross_tab.c.indptr[i] as usize;
            let hi = cross_tab.c.indptr[i + 1] as usize;
            let row_abs: f64 = (lo..hi)
                .map(|idx| {
                    let j = cross_tab.c.indices[idx] as usize;
                    (d[i] * d[n_q + j] * cross_tab.c.data[idx]).abs()
                })
                .sum();
            let scaled_diag = d[i] * d[i] * diagonals.q[i];
            assert!(
                scaled_diag * (1.0 + 2.0 * WDD_SLACK) >= row_abs,
                "q-row {i}: scaled diag {scaled_diag} < row abs {row_abs}"
            );
        }
    }

    fn assert_folds_non_negative(cross_tab: &CrossTab, d: &[f64]) {
        let n_q = cross_tab.n_q();
        for i in 0..n_q {
            let lo = cross_tab.c.indptr[i] as usize;
            let hi = cross_tab.c.indptr[i + 1] as usize;
            for idx in lo..hi {
                let j = cross_tab.c.indices[idx] as usize;
                let v = d[i] * d[n_q + j] * cross_tab.c.data[idx];
                assert!(v >= 0.0, "cell ({i},{j}) folds to {v}");
            }
        }
    }

    #[test]
    fn scalable_balanced_component_folds_non_negative_and_reaches_wdd() {
        // Backward-constructed from a strictly-SDD plain Â and mixed-sign d,
        // so a dominant scaling exists and the relaxation must find one.
        let (n_q, n_r) = (2usize, 3usize);
        let d_true = [2.0, -0.5, -1.0, 4.0, 0.25];
        let c_hat = [[1.0, 2.0, 0.5], [3.0, 0.0, 1.5]];
        let diag_hat = [4.0, 5.0, 4.5, 2.5, 2.375];

        let mut c_raw = vec![0.0; n_q * n_r];
        for i in 0..n_q {
            for j in 0..n_r {
                c_raw[i * n_r + j] = c_hat[i][j] / (d_true[i] * d_true[n_q + j]);
            }
        }
        let ct = cross_tab_of(&c_raw, n_q, n_r);
        let diagonals = BlockDiagonals {
            q: (0..n_q)
                .map(|k| diag_hat[k] / (d_true[k] * d_true[k]))
                .collect(),
            r: (n_q..n_q + n_r)
                .map(|k| diag_hat[k] / (d_true[k] * d_true[k]))
                .collect(),
        };

        let d = balance_and_scale(&ct, &diagonals)
            .expect("balanced component")
            .expect("mixed signs need a congruence");
        assert_folds_non_negative(&ct, &d);
        assert_wdd(&ct, &diagonals, &d);
    }

    #[test]
    fn already_wdd_mixed_sign_component_gets_a_pure_signature() {
        let ct = cross_tab_of(&[1.0, -1.0, 2.0, -2.0], 2, 2);
        let diagonals = BlockDiagonals {
            q: vec![3.0, 5.0],
            r: vec![4.0, 6.0],
        };
        let d = balance_and_scale(&ct, &diagonals)
            .expect("balanced component")
            .expect("mixed signs need a congruence");
        assert_eq!(&*d, &[1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn frustrated_four_cycle_errors() {
        // Sign product around the q0-r0-q1-r1 cycle is negative.
        let ct = cross_tab_of(&[1.0, 1.0, 1.0, -1.0], 2, 2);
        let diagonals = BlockDiagonals {
            q: vec![2.0, 2.0],
            r: vec![2.0, 2.0],
        };
        assert!(balance_and_scale(&ct, &diagonals).is_err());
    }

    #[test]
    fn exactly_singular_boundary_is_accepted_within_slack() {
        // Congruence-scaled singular boundary (kernel [1/2, −1, −1]): the
        // exact fixed point has t_i == μ_i, so only the slack lets the
        // relaxation certify and exit instead of chasing rounding.
        let ct = cross_tab_of(&[0.5, -1.0], 2, 1);
        let diagonals = BlockDiagonals {
            q: vec![0.25, 1.0],
            r: vec![2.0],
        };
        let d = balance_and_scale(&ct, &diagonals)
            .expect("balanced component")
            .expect("not WDD at λ ≡ 1");
        assert_folds_non_negative(&ct, &d);
        assert_wdd(&ct, &diagonals, &d);
    }

    #[test]
    fn non_scalable_component_exhausts_budget_with_finite_d() {
        // ρ of the normalized structure exceeds 1: no dominant scaling
        // exists, μ grows every sweep, and the budget hands back a finite d.
        let ct = cross_tab_of(&[1.0, -1.0, 2.0, -2.0], 2, 2);
        let diagonals = BlockDiagonals {
            q: vec![1.0, 2.0],
            r: vec![1.0, 2.0],
        };
        let d = balance_and_scale(&ct, &diagonals)
            .expect("balanced component")
            .expect("non-WDD component needs a congruence");
        assert!(d.iter().all(|v| v.is_finite()));
        assert_folds_non_negative(&ct, &d);
    }
}
