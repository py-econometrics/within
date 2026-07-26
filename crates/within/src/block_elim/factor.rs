//! Reduced-system factor backends for Schur-complement local solves.
//!
//! [`ReducedSystem`] pairs an approximate sparse or dense direct factor with
//! the floating, grounded, or signed solve semantics it supports.
//! [`factor_sparse`] bridges to the `approx-chol` builder, surfacing build errors
//! as [`BuildError::LocalSolverBuild`].

use approx_chol::low_level::Builder;
use approx_chol::{CsrRef, Factor};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::{MatMut, MatRef, Par, Side};
use schwarz_precond::LocalSolveError;

use super::csr_matrix::CsrMatrix;
use crate::config::ApproxCholConfig;
use crate::BuildError;

// ===========================================================================
// ReducedSystem — factor backend paired with its solve semantics
// ===========================================================================

#[derive(Clone, serde::Serialize)]
pub(crate) enum DirectFactor {
    Approx(Factor),
    Dense(DenseCholesky),
}

impl<'de> serde::Deserialize<'de> for DirectFactor {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        enum Repr {
            Approx(Factor),
            Dense(DenseCholesky),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Approx(factor) => checked_approx(factor),
            Repr::Dense(factor) => checked_dense(factor),
        }
        .map_err(D::Error::custom)
    }
}

impl DirectFactor {
    pub(crate) fn try_dense(minor: Vec<f64>, n: usize) -> Option<Self> {
        DenseCholesky::try_factor(minor, n).map(Self::Dense)
    }

    pub(crate) fn input_dimension(&self) -> usize {
        match self {
            Self::Approx(f) => f.original_n(),
            Self::Dense(f) => f.n(),
        }
    }

    pub(crate) fn factor_dimension(&self) -> usize {
        match self {
            Self::Approx(f) => f.n(),
            Self::Dense(f) => f.n(),
        }
    }

    fn solve_in_place(&self, x: &mut [f64]) -> Result<(), LocalSolveError> {
        match self {
            Self::Approx(f) => {
                debug_assert_eq!(f.n(), x.len());
                f.solve_in_place(x)
                    .map_err(|e| LocalSolveError::BackendFailed {
                        context: "within.local.block_elim.reduced_approx",
                        message: e.to_string(),
                    })?;
                Ok(())
            }
            Self::Dense(f) => {
                f.solve_in_place(x);
                Ok(())
            }
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct CoverFactor {
    inner: DirectFactor,
    m: usize,
}

impl<'de> serde::Deserialize<'de> for CoverFactor {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        struct Repr {
            inner: DirectFactor,
            m: usize,
        }

        let repr = Repr::deserialize(deserializer)?;
        Self::try_new(repr.inner, repr.m).map_err(D::Error::custom)
    }
}

impl CoverFactor {
    pub(crate) fn try_new(inner: DirectFactor, m: usize) -> Result<Self, &'static str> {
        let two_m = m.checked_mul(2).ok_or("Cover dimension m too large")?;
        let cover_dimension = inner.factor_dimension();
        if cover_dimension < two_m || cover_dimension > two_m.saturating_add(2) {
            return Err("Cover inner factor dimension inconsistent with m");
        }
        Ok(Self { inner, m })
    }

    fn solve_in_place(&self, x: &mut [f64], scratch: &mut [f64]) -> Result<(), LocalSolveError> {
        debug_assert_eq!(self.m, x.len());
        let cover_n = self.inner.factor_dimension();
        let buf = &mut scratch[..cover_n];
        buf[..self.m].copy_from_slice(x);
        for (out, &value) in buf[self.m..2 * self.m].iter_mut().zip(x.iter()) {
            *out = -value;
        }
        buf[2 * self.m..].fill(0.0);
        self.inner.solve_in_place(buf)?;
        for (i, out) in x.iter_mut().enumerate() {
            *out = 0.5 * (buf[i] - buf[self.m + i]);
        }
        Ok(())
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum ReducedSystem {
    Floating(DirectFactor),
    Grounded(DirectFactor),
    Signed(CoverFactor),
}

impl ReducedSystem {
    pub(crate) fn factor_dimension(&self) -> usize {
        match self {
            Self::Floating(factor) | Self::Grounded(factor) => factor.factor_dimension(),
            Self::Signed(factor) => factor.m,
        }
    }

    pub(crate) fn scratch_len(&self) -> usize {
        match self {
            Self::Floating(_) | Self::Grounded(_) => 0,
            Self::Signed(factor) => factor.inner.factor_dimension(),
        }
    }

    pub(crate) fn solve_in_place(
        &self,
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), LocalSolveError> {
        match self {
            Self::Floating(factor) | Self::Grounded(factor) => factor.solve_in_place(x),
            Self::Signed(factor) => factor.solve_in_place(x, scratch),
        }
    }

    pub(crate) fn validate_keep_dimension(&self, n_keep: usize) -> Result<(), &'static str> {
        match self {
            Self::Floating(factor) if factor.input_dimension() == n_keep => Ok(()),
            Self::Grounded(factor)
                if factor.input_dimension() == n_keep
                    || factor.input_dimension() == n_keep.saturating_add(1) =>
            {
                Ok(())
            }
            Self::Signed(factor) if factor.m == n_keep => Ok(()),
            _ => Err("reduced system dimension disagrees with kept block"),
        }
    }
}

fn checked_approx(factor: Factor) -> Result<DirectFactor, &'static str> {
    let (n, base) = (factor.n(), factor.original_n());
    if n == base || Some(n) == base.checked_add(1) {
        Ok(DirectFactor::Approx(factor))
    } else {
        Err("Approx factor dimension inconsistent with its original dimension")
    }
}

fn checked_dense(factor: DenseCholesky) -> Result<DirectFactor, &'static str> {
    let n = factor.n;
    let full = n.checked_mul(n);
    let anchored = n.saturating_sub(1).checked_mul(n.saturating_sub(1));
    if Some(factor.l_row_major.len()) == full || Some(factor.l_row_major.len()) == anchored {
        Ok(DirectFactor::Dense(factor))
    } else {
        Err("DenseCholesky factor length inconsistent with its dimension")
    }
}

// ===========================================================================
// DenseCholesky — dense Cholesky on a principal minor
// ===========================================================================

/// Dense Cholesky on a principal `m × m` minor of the reduced Schur system.
///
/// `m == n` factors the full (nonsingular) complement; `m == n − 1` anchors
/// the last node (`x = 0`) — the gauge for a floating Laplacian.
/// `m` is recovered from the factor length, keeping the wire format unchanged.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DenseCholesky {
    /// Lower-triangular factor of the `m × m` minor, row-major.
    l_row_major: Vec<f64>,
    /// Full Schur dimension.
    n: usize,
}

impl DenseCholesky {
    fn try_factor(minor: Vec<f64>, n: usize) -> Option<Self> {
        let anchored = n.saturating_sub(1);
        let m = match minor.len() {
            len if len == n * n => n,
            len if len == anchored * anchored => anchored,
            _ => return None,
        };
        if m == 0 {
            return Some(Self {
                l_row_major: Vec::new(),
                n,
            });
        }

        let mat_ref = MatRef::from_row_major_slice(&minor, m, m);
        let llt = mat_ref.llt(Side::Lower).ok()?;
        let l = llt.L();

        let mut l_row_major = vec![0.0; m * m];
        for r in 0..m {
            for c in 0..=r {
                l_row_major[r * m + c] = l[(r, c)];
            }
        }

        Some(Self { l_row_major, n })
    }

    fn n(&self) -> usize {
        self.n
    }

    /// Solve `L L^T x = b` on the factored minor in-place.
    ///
    /// Expects `x.len() == n`; coordinates past the minor (`x[m..]`) are the
    /// anchored gauge and are written as zero.
    fn solve_in_place(&self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        let m = if self.l_row_major.len() == self.n * self.n {
            self.n
        } else {
            self.n.saturating_sub(1)
        };
        debug_assert_eq!(self.l_row_major.len(), m * m);

        if m > 0 {
            // Delegate both substitutions to faer's blocked kernels, viewing
            // the stored factor and `x` in place: L y = b then Lᵀ x = y.
            let l = MatRef::from_row_major_slice(&self.l_row_major, m, m);
            solve_lower_triangular_in_place(
                l,
                MatMut::from_column_major_slice_mut(&mut x[..m], m, 1),
                Par::Seq,
            );
            solve_upper_triangular_in_place(
                l.transpose(),
                MatMut::from_column_major_slice_mut(&mut x[..m], m, 1),
                Par::Seq,
            );
        }

        for v in &mut x[m..] {
            *v = 0.0;
        }
    }
}

// ===========================================================================
// factor_sparse — bridge into approx_chol for sparse reduced Schur
// ===========================================================================

/// Factor a sparse (SDDM) reduced Schur complement into a direct `Approx` factor.
pub(crate) fn factor_sparse(
    matrix: &CsrMatrix,
    approx_chol: ApproxCholConfig,
) -> Result<DirectFactor, BuildError> {
    let schur_builder = Builder::new(approx_chol.to_approx_chol());
    let csr = CsrRef::new(
        matrix.indptr(),
        matrix.indices(),
        matrix.data(),
        u32::try_from(matrix.n()).expect("Schur complement dimension exceeds u32::MAX"),
    )
    .map_err(|e| BuildError::LocalSolverBuild(format!("invalid Schur complement CSR: {e}")))?;
    schur_builder
        .build(csr)
        .map(DirectFactor::Approx)
        .map_err(|e| {
            BuildError::LocalSolverBuild(format!("failed Schur complement factorization: {e}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_cholesky_solve_n0() {
        let chol = DenseCholesky {
            l_row_major: Vec::new(),
            n: 0,
        };
        let mut x: Vec<f64> = Vec::new();
        chol.solve_in_place(&mut x); // should be no-op
    }

    #[test]
    fn test_dense_cholesky_solve_n1_anchored() {
        let chol = DenseCholesky {
            l_row_major: Vec::new(),
            n: 1,
        };
        let mut x = vec![42.0];
        chol.solve_in_place(&mut x);
        assert_eq!(x[0], 0.0); // empty anchored minor -> x[0] = 0
    }

    #[test]
    fn test_try_factor_wrong_length() {
        // n=3 admits minors of 9 (full) or 4 (anchored) elements; give 3.
        let result = DenseCholesky::try_factor(vec![1.0, 2.0, 3.0], 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_factor_anchored_n1() {
        // n=1, empty minor -> anchored m=0, Some with empty factor.
        let result = DenseCholesky::try_factor(Vec::new(), 1);
        assert!(result.is_some());
        assert_eq!(result.unwrap().n(), 1);
    }

    #[test]
    fn test_try_factor_singular() {
        // Singular 2x2 anchored minor of n=3 (both rows identical).
        let result = DenseCholesky::try_factor(vec![1.0, 1.0, 1.0, 1.0], 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_full_minor_solves_directly() {
        // Full 2x2 SPD minor (m == n): plain LLT solve, no anchored zero.
        let chol = DenseCholesky::try_factor(vec![4.0, 0.0, 0.0, 9.0], 2).unwrap();
        let mut x = vec![8.0, 18.0];
        chol.solve_in_place(&mut x);
        assert_eq!(x, vec![2.0, 2.0]);
    }

    #[test]
    fn cover_reduced_factor_solves_signed_system() {
        // Gremban double cover of the signed M = [[2, 1], [1, 2]] (a positive,
        // frustrated off-diagonal). Node order [0, 1, 0', 1']; the positive
        // edge (0,1) routes to 0-1' and 0'-1 as SDDM off-diagonals, so the
        // cover is strictly diagonally dominant. Verified by hand that
        // Ŝ·[z, -z] = [Mz, -Mz], hence the antisymmetric embed of b recovers
        // M⁻¹b — this pins the cover embed/read-back, not
        // approx-chol's accuracy (checked via the residual M x ≈ b).
        let cover = CsrMatrix::new(
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0],
            4,
        );
        let inner = factor_sparse(
            &cover,
            ApproxCholConfig {
                seed: 0,
                split_merge: Some(2),
            },
        )
        .expect("factor cover");
        let reduced =
            ReducedSystem::Signed(CoverFactor::try_new(inner, 2).expect("valid cover dimensions"));
        assert_eq!(reduced.factor_dimension(), 2);

        let b = [1.0, 0.5];
        let mut x = b;
        let mut scratch = vec![0.0; reduced.scratch_len()];
        reduced
            .solve_in_place(&mut x, &mut scratch)
            .expect("cover solve");

        // Residual of the signed system M x = b (M = [[2,1],[1,2]]).
        let r0 = 2.0 * x[0] + x[1] - b[0];
        let r1 = x[0] + 2.0 * x[1] - b[1];
        assert!(r0.hypot(r1) < 1e-9, "residual too large: ({r0}, {r1})");
    }

    fn dense_2x2() -> DirectFactor {
        DirectFactor::try_dense(vec![4.0, 0.0, 0.0, 9.0], 2).expect("spd minor")
    }

    #[test]
    fn valid_dense_round_trips() {
        let system = ReducedSystem::Floating(dense_2x2());
        let bytes = postcard::to_stdvec(&system).expect("serialize");
        let restored: ReducedSystem = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.factor_dimension(), 2);
    }

    #[test]
    fn dense_with_inconsistent_length_is_rejected() {
        #[allow(dead_code)]
        #[derive(serde::Serialize)]
        enum UncheckedDirectFactor {
            Approx(Factor),
            Dense(DenseCholesky),
        }

        let bad = UncheckedDirectFactor::Dense(DenseCholesky {
            l_row_major: vec![1.0, 2.0, 3.0],
            n: 4,
        });
        let bytes = postcard::to_stdvec(&bad).expect("serialize");
        assert!(postcard::from_bytes::<DirectFactor>(&bytes).is_err());
    }

    #[test]
    fn cover_with_undersized_inner_is_rejected() {
        #[derive(serde::Serialize)]
        struct UncheckedCoverFactor {
            inner: DirectFactor,
            m: usize,
        }

        let bad = UncheckedCoverFactor {
            inner: dense_2x2(),
            m: 5,
        };
        let bytes = postcard::to_stdvec(&bad).expect("serialize");
        assert!(postcard::from_bytes::<CoverFactor>(&bytes).is_err());
    }
}
