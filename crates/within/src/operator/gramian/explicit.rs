use std::sync::Arc;

use schwarz_precond::{Operator, SparseMatrix};

use super::Gramian;
use crate::domain::{build_gramian_blocks, Design, PairBlockData};
use crate::observation::{FactorMeta, Store};
use crate::{WithinError, WithinResult};

// ===========================================================================
// DENSE_TABLE_MAX_ENTRIES constant
// ===========================================================================

/// Max entries in a flat dense table (~40 MB at 8 bytes each).
pub(super) const DENSE_TABLE_MAX_ENTRIES: usize = 5_000_000;

// ===========================================================================
// Gramian — explicit CSR construction
// ===========================================================================

impl Gramian {
    /// Assemble the Gramian `G = D^T W D` as a CSR sparse matrix.
    ///
    /// Pass `weights = None` for the unweighted form `G = D^T D`.
    pub fn build<S: Store>(design: &Design<S>, weights: Option<&[f64]>) -> Self {
        if let Some(w) = weights {
            assert_eq!(
                w.len(),
                design.n_rows,
                "weights length {} does not match design.n_rows {}",
                w.len(),
                design.n_rows
            );
        }
        // Single-factor designs have no factor pairs; the Gramian is the level-count diagonal.
        if design.n_factors() < 2 {
            return Self::build_diagonal_only(design, weights);
        }
        let blocks = build_gramian_blocks(design, weights);
        Self::from_pair_blocks(&blocks, &design.factors, design.n_dofs)
            .expect("Gramian assembly overflow")
    }

    /// Assemble a diagonal-only Gramian for designs with no factor pairs (≤1 factor).
    fn build_diagonal_only<S: Store>(design: &Design<S>, weights: Option<&[f64]>) -> Self {
        let n_dofs = design.n_dofs;
        let n_factors = design.n_factors();

        let mut diag = vec![0.0f64; n_dofs];
        for uid in 0..design.store.n_obs() {
            let w = weights.map_or(1.0, |w| w[uid]);
            for q in 0..n_factors {
                let idx = design.factors[q].offset + design.store.level(uid, q) as usize;
                diag[idx] += w;
            }
        }

        let mut indptr = vec![0u32; n_dofs + 1];
        for i in 0..n_dofs {
            indptr[i + 1] = indptr[i] + if diag[i] > 0.0 { 1 } else { 0 };
        }
        let nnz = indptr[n_dofs] as usize;
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for (i, &d) in diag.iter().enumerate() {
            if d > 0.0 {
                indices.push(i as u32);
                data.push(d);
            }
        }

        Self {
            matrix: Arc::new(SparseMatrix::new(indptr, indices, data, n_dofs)),
        }
    }

    /// Compose the full Gramian CSR from pre-built per-pair block data.
    pub(crate) fn from_pair_blocks(
        blocks: &[PairBlockData],
        factors: &[FactorMeta],
        n_dofs: usize,
    ) -> WithinResult<Self> {
        Ok(Self {
            matrix: Arc::new(compose_gramian_from_blocks(blocks, factors, n_dofs)?),
        })
    }

    /// Matrix-vector product `y = G x`.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.matrix.matvec(x, y);
    }

    /// Diagonal of `G`.
    pub fn diagonal(&self) -> Vec<f64> {
        self.matrix.diagonal()
    }

    /// Extract submatrix `G[indices, indices]`.
    pub fn extract_submatrix(&self, indices: &[usize]) -> SparseMatrix {
        self.matrix.extract_submatrix(indices)
    }

    /// Number of DOFs.
    pub fn n_dofs(&self) -> usize {
        self.matrix.n()
    }
}

impl Operator for Gramian {
    fn nrows(&self) -> usize {
        self.n_dofs()
    }

    fn ncols(&self) -> usize {
        self.n_dofs()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.matvec(x, y);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.apply(x, y)
    }
}

/// Compose a contiguous Gramian CSR from per-pair block data.
fn compose_gramian_from_blocks(
    blocks: &[PairBlockData],
    factors: &[FactorMeta],
    n_dofs: usize,
) -> WithinResult<SparseMatrix> {
    let n_factors = factors.len();

    // Collect canonical diagonal per factor (from any block involving that factor).
    let mut factor_diag: Vec<Option<&[f64]>> = vec![None; n_factors];
    let mut factor_global: Vec<Option<&[u32]>> = vec![None; n_factors];
    for b in blocks {
        if factor_diag[b.q].is_none() {
            factor_diag[b.q] = Some(&b.cross_tab.diag_q);
            factor_global[b.q] = Some(&b.q_global);
        }
        if factor_diag[b.r].is_none() {
            factor_diag[b.r] = Some(&b.cross_tab.diag_r);
            factor_global[b.r] = Some(&b.r_global);
        }
    }

    // Group blocks by factor for fast lookup.
    let mut first_pairs: Vec<Vec<usize>> = vec![Vec::new(); n_factors];
    let mut second_pairs: Vec<Vec<usize>> = vec![Vec::new(); n_factors];
    for (bi, b) in blocks.iter().enumerate() {
        first_pairs[b.q].push(bi);
        second_pairs[b.r].push(bi);
    }

    // --- Pass 1: count NNZ per row ---
    let mut row_nnz = vec![0u64; n_dofs];

    for f_global in factor_global.iter().flatten() {
        for &g in *f_global {
            row_nnz[g as usize] += 1;
        }
    }

    for b in blocks {
        for (cj, &g) in b.q_global.iter().enumerate() {
            let nnz_in_row = (b.cross_tab.c.indptr[cj + 1] - b.cross_tab.c.indptr[cj]) as u64;
            row_nnz[g as usize] += nnz_in_row;
        }
        for (ck, &g) in b.r_global.iter().enumerate() {
            let nnz_in_row = (b.cross_tab.ct.indptr[ck + 1] - b.cross_tab.ct.indptr[ck]) as u64;
            row_nnz[g as usize] += nnz_in_row;
        }
    }

    // --- Build indptr ---
    let mut indptr = vec![0u32; n_dofs + 1];
    for i in 0..n_dofs {
        let nnz_u64 = row_nnz[i];
        indptr[i + 1] =
            indptr[i]
                .checked_add(u32::try_from(nnz_u64).map_err(|_| {
                    WithinError::Overflow(format!("row nnz exceeds u32 at row {i}"))
                })?)
                .ok_or_else(|| {
                    WithinError::Overflow(format!("cumulative indptr exceeds u32 at row {i}"))
                })?;
    }
    let total_nnz = indptr[n_dofs] as usize;

    // --- Pass 2: fill indices and data ---
    let mut indices = vec![0u32; total_nnz];
    let mut data = vec![0.0f64; total_nnz];
    let mut cursor = indptr[..n_dofs].to_vec();

    for f in 0..n_factors {
        let f_global = match factor_global[f] {
            Some(g) => g,
            None => continue,
        };
        let f_diag = factor_diag[f].unwrap();
        let n_active_f = f_global.len();

        for compact_j in 0..n_active_f {
            let g = f_global[compact_j] as usize;

            // 1. Entries from pairs (p, f) where p < f
            for &bi in &second_pairs[f] {
                let b = &blocks[bi];
                let start = b.cross_tab.ct.indptr[compact_j] as usize;
                let end = b.cross_tab.ct.indptr[compact_j + 1] as usize;
                for idx in start..end {
                    let compact_p = b.cross_tab.ct.indices[idx] as usize;
                    debug_assert!(
                        compact_p < b.q_global.len(),
                        "compact_p {compact_p} out of range for q_global len {}",
                        b.q_global.len()
                    );
                    let pos = cursor[g] as usize;
                    indices[pos] = b.q_global[compact_p];
                    data[pos] = b.cross_tab.ct.data[idx];
                    cursor[g] += 1;
                }
            }

            // 2. Diagonal
            let pos = cursor[g] as usize;
            indices[pos] = g as u32;
            data[pos] = f_diag[compact_j];
            cursor[g] += 1;

            // 3. Entries from pairs (f, r) where r > f
            for &bi in &first_pairs[f] {
                let b = &blocks[bi];
                let start = b.cross_tab.c.indptr[compact_j] as usize;
                let end = b.cross_tab.c.indptr[compact_j + 1] as usize;
                for idx in start..end {
                    let compact_r = b.cross_tab.c.indices[idx] as usize;
                    let pos = cursor[g] as usize;
                    indices[pos] = b.r_global[compact_r];
                    data[pos] = b.cross_tab.c.data[idx];
                    cursor[g] += 1;
                }
            }
        }
    }

    Ok(SparseMatrix::new(indptr, indices, data, n_dofs))
}
