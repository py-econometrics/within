use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use schwarz_precond::{Operator, SparseMatrix};

use super::Gramian;
use crate::domain::{PairBlockData, WeightedDesign};
use crate::observation::{FactorMeta, ObservationStore};

// ===========================================================================
// CSR assembly helper
// ===========================================================================

/// Build a symmetric CSR matrix from (row, col, value) entries.
///
/// `emit_entries` is called twice with a callback:
///   1. Counting pass — the callback tallies row counts
///   2. Fill pass — the callback writes (col, value) into CSR arrays
///
/// After filling, each row is sorted by column index (insertion sort).
fn build_symmetric_csr(
    n: usize,
    emit_entries: impl Fn(&mut dyn FnMut(usize, usize, f64)),
) -> SparseMatrix {
    // Pass 1: count NNZ per row
    let mut row_counts = vec![0u32; n];
    emit_entries(&mut |row, _col, _val| {
        row_counts[row] += 1;
    });

    // Build indptr via prefix sum
    let mut indptr = vec![0u32; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + row_counts[i];
    }
    let nnz = indptr[n] as usize;

    // Pass 2: fill indices + data using indptr copy as write cursors
    let mut cursor = indptr[..n].to_vec();
    let mut indices = vec![0u32; nnz];
    let mut data = vec![0.0f64; nnz];
    emit_entries(&mut |row, col, val| {
        let pos = cursor[row] as usize;
        indices[pos] = col as u32;
        data[pos] = val;
        cursor[row] += 1;
    });

    // Sort indices within each row by column index.
    // Use indirect permutation sort for large rows to avoid O(n^2) insertion sort.
    for row in 0..n {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let len = end - start;
        if len <= 1 {
            continue;
        }
        let row_idx = &mut indices[start..end];
        let row_data = &mut data[start..end];
        if len <= 64 {
            // Insertion sort for small rows (cache-friendly, low overhead)
            for i in 1..len {
                let key_col = row_idx[i];
                let key_val = row_data[i];
                let mut j = i;
                while j > 0 && row_idx[j - 1] > key_col {
                    row_idx[j] = row_idx[j - 1];
                    row_data[j] = row_data[j - 1];
                    j -= 1;
                }
                row_idx[j] = key_col;
                row_data[j] = key_val;
            }
        } else {
            // Build permutation, sort it, then apply in-place
            let mut perm: Vec<u32> = (0..len as u32).collect();
            perm.sort_unstable_by_key(|&i| row_idx[i as usize]);
            // Apply permutation to both arrays via cycle-following
            let mut visited = vec![false; len];
            for i in 0..len {
                if visited[i] || perm[i] as usize == i {
                    continue;
                }
                let saved_idx = row_idx[i];
                let saved_data = row_data[i];
                let mut j = i;
                loop {
                    visited[j] = true;
                    let src = perm[j] as usize;
                    if src == i {
                        row_idx[j] = saved_idx;
                        row_data[j] = saved_data;
                        break;
                    }
                    row_idx[j] = row_idx[src];
                    row_data[j] = row_data[src];
                    j = src;
                }
            }
        }
    }

    SparseMatrix::new(indptr, indices, data, n)
}

// ===========================================================================
// DENSE_TABLE_MAX_ENTRIES constant
// ===========================================================================

/// Max entries in a flat dense table (~40 MB at 8 bytes each).
pub(super) const DENSE_TABLE_MAX_ENTRIES: usize = 5_000_000;

// ===========================================================================
// PairAccumulator — weighted count accumulator for factor-pair cross tables
// ===========================================================================

enum PairStorage {
    Dense(Vec<f64>),
    Sparse(HashMap<usize, f64>),
}

/// Accumulates weighted counts for a factor-pair cross table.
pub(super) struct PairAccumulator {
    n_rows: usize,
    n_cols: usize,
    storage: PairStorage,
}

impl PairAccumulator {
    pub(super) fn new(n_rows: usize, n_cols: usize, n_unique: usize) -> Self {
        let table_size = n_rows * n_cols;
        let storage = if table_size <= DENSE_TABLE_MAX_ENTRIES {
            PairStorage::Dense(vec![0.0; table_size])
        } else {
            PairStorage::Sparse(HashMap::with_capacity(n_unique.min(table_size)))
        };
        Self {
            n_rows,
            n_cols,
            storage,
        }
    }

    #[inline]
    pub(super) fn add(&mut self, row: usize, col: usize, weight: f64) {
        let key = row * self.n_cols + col;
        match &mut self.storage {
            PairStorage::Dense(table) => {
                table[key] += weight;
            }
            PairStorage::Sparse(table) => {
                *table.entry(key).or_insert(0.0) += weight;
            }
        }
    }

    fn merge_from(&mut self, other: &PairAccumulator) {
        match (&mut self.storage, &other.storage) {
            (PairStorage::Dense(a), PairStorage::Dense(b)) => {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += *bi;
                }
            }
            (PairStorage::Sparse(a), PairStorage::Sparse(b)) => {
                for (&k, &v) in b {
                    *a.entry(k).or_insert(0.0) += v;
                }
            }
            _ => unreachable!("storage type determined by factor dimensions, same for all threads"),
        }
    }

    pub(super) fn for_each_nonzero(&self, mut f: impl FnMut(usize, usize, f64)) {
        match &self.storage {
            PairStorage::Dense(table) => {
                for row in 0..self.n_rows {
                    for col in 0..self.n_cols {
                        let v = table[row * self.n_cols + col];
                        if v > 0.0 {
                            f(row, col, v);
                        }
                    }
                }
            }
            PairStorage::Sparse(table) => {
                for (&key, &v) in table {
                    if v > 0.0 {
                        f(key / self.n_cols, key % self.n_cols, v);
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Gramian — explicit CSR construction
// ===========================================================================

impl Gramian {
    pub fn build<S: ObservationStore>(design: &WeightedDesign<S>) -> Self {
        Self {
            matrix: Arc::new(build_full_matrix(design)),
        }
    }

    /// Build a Gramian containing only the single factor pair `(q, r)`.
    pub fn build_for_pair<S: ObservationStore>(
        design: &WeightedDesign<S>,
        q: usize,
        r: usize,
    ) -> Self {
        Self {
            matrix: Arc::new(build_pair_matrix(design, q, r)),
        }
    }

    /// Compose the full Gramian CSR from pre-built per-pair block data.
    ///
    /// Each `PairBlockData` carries the off-diagonal block C_qr (and C_qr^T),
    /// diagonal entries for both factors, and the local-to-global index mapping.
    /// Because factor offsets are monotonically increasing, column indices within
    /// each row are emitted in sorted order — no per-row sorting is needed.
    pub(crate) fn from_pair_blocks(
        blocks: &[PairBlockData],
        factors: &[FactorMeta],
        n_dofs: usize,
    ) -> Self {
        Self {
            matrix: Arc::new(compose_gramian_from_blocks(blocks, factors, n_dofs)),
        }
    }

    /// `G @ x`.
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

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.matvec(x, y);
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        self.apply(x, y);
    }
}

fn build_full_matrix<S: ObservationStore>(design: &WeightedDesign<S>) -> SparseMatrix {
    let n_dofs = design.n_dofs;
    let n_obs = design.store.n_obs();
    let n_factors = design.n_factors();

    let pair_info: Vec<(usize, usize)> = {
        let mut v = Vec::with_capacity(n_factors * (n_factors - 1) / 2);
        for q in 0..n_factors {
            for r in (q + 1)..n_factors {
                v.push((q, r));
            }
        }
        v
    };

    const PAR_THRESHOLD: usize = 100_000;

    let (diag_flat, pair_tables) = if n_obs > PAR_THRESHOLD {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = n_obs.div_ceil(n_threads);

        let partials: Vec<(Vec<f64>, Vec<PairAccumulator>)> = (0..n_threads)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(n_obs);
                let chunk_len = end.saturating_sub(start);

                let mut diag = vec![0.0f64; n_dofs];
                let mut pairs: Vec<PairAccumulator> = pair_info
                    .iter()
                    .map(|&(q, r)| {
                        PairAccumulator::new(
                            design.factors[q].n_levels,
                            design.factors[r].n_levels,
                            chunk_len,
                        )
                    })
                    .collect();

                for uid in start..end {
                    let w = design.uid_weight(uid);
                    for q in 0..n_factors {
                        diag[design.factors[q].offset + design.store.level(uid, q) as usize] += w;
                    }
                    for (pi, &(q, r)) in pair_info.iter().enumerate() {
                        pairs[pi].add(
                            design.store.level(uid, q) as usize,
                            design.store.level(uid, r) as usize,
                            w,
                        );
                    }
                }

                (diag, pairs)
            })
            .collect();

        // Merge thread-local results
        let mut iter = partials.into_iter();
        let (mut diag_flat, mut pair_tables) = iter.next().unwrap();
        for (d, p) in iter {
            for (a, b) in diag_flat.iter_mut().zip(d.iter()) {
                *a += *b;
            }
            for (pi, pair_table) in pair_tables.iter_mut().enumerate() {
                pair_table.merge_from(&p[pi]);
            }
        }
        (diag_flat, pair_tables)
    } else {
        let mut diag_flat = vec![0.0f64; n_dofs];
        let mut pair_tables: Vec<PairAccumulator> = pair_info
            .iter()
            .map(|&(q, r)| {
                PairAccumulator::new(
                    design.factors[q].n_levels,
                    design.factors[r].n_levels,
                    n_obs,
                )
            })
            .collect();

        for uid in 0..n_obs {
            let w = design.uid_weight(uid);
            for q in 0..n_factors {
                diag_flat[design.factors[q].offset + design.store.level(uid, q) as usize] += w;
            }
            for (pi, &(q, r)) in pair_info.iter().enumerate() {
                pair_tables[pi].add(
                    design.store.level(uid, q) as usize,
                    design.store.level(uid, r) as usize,
                    w,
                );
            }
        }
        (diag_flat, pair_tables)
    };

    build_symmetric_csr(n_dofs, |emit| {
        for (row, &cnt) in diag_flat.iter().enumerate() {
            if cnt > 0.0 {
                emit(row, row, cnt);
            }
        }

        for (pi, &(q, r)) in pair_info.iter().enumerate() {
            let fq = &design.factors[q];
            let fr = &design.factors[r];
            pair_tables[pi].for_each_nonzero(|j, k, cnt| {
                let gj = fq.offset + j;
                let gk = fr.offset + k;
                emit(gj, gk, cnt);
                emit(gk, gj, cnt);
            });
        }
    })
}

fn build_pair_matrix<S: ObservationStore>(
    design: &WeightedDesign<S>,
    q: usize,
    r: usize,
) -> SparseMatrix {
    let n_dofs = design.n_dofs;
    let n_obs = design.store.n_obs();
    let fq = &design.factors[q];
    let fr = &design.factors[r];

    let mut diag_q = vec![0.0; fq.n_levels];
    let mut diag_r = vec![0.0; fr.n_levels];
    let mut table = PairAccumulator::new(fq.n_levels, fr.n_levels, n_obs);

    for uid in 0..n_obs {
        let w = design.uid_weight(uid);
        let j = design.store.level(uid, q) as usize;
        let k = design.store.level(uid, r) as usize;
        diag_q[j] += w;
        diag_r[k] += w;
        table.add(j, k, w);
    }

    build_symmetric_csr(n_dofs, |emit| {
        for (j, &cnt) in diag_q.iter().enumerate() {
            if cnt > 0.0 {
                let gj = fq.offset + j;
                emit(gj, gj, cnt);
            }
        }
        for (k, &cnt) in diag_r.iter().enumerate() {
            if cnt > 0.0 {
                let gk = fr.offset + k;
                emit(gk, gk, cnt);
            }
        }

        table.for_each_nonzero(|j, k, cnt| {
            let gj = fq.offset + j;
            let gk = fr.offset + k;
            emit(gj, gk, cnt);
            emit(gk, gj, cnt);
        });
    })
}

/// Compose a contiguous Gramian CSR from per-pair block data.
///
/// The Gramian has block structure:
/// ```text
/// G = | D_0    C_01   C_02   ... |
///     | C_01^T D_1    C_12   ... |
///     | C_02^T C_12^T D_2    ... |
/// ```
///
/// Since factor offsets are monotonically increasing, entries within each row
/// are emitted in sorted column order: pairs with earlier factors first,
/// then the diagonal, then pairs with later factors. No per-row sorting needed.
fn compose_gramian_from_blocks(
    blocks: &[PairBlockData],
    factors: &[FactorMeta],
    n_dofs: usize,
) -> SparseMatrix {
    let n_factors = factors.len();

    // Collect canonical diagonal per factor (from any block involving that factor).
    let mut factor_diag: Vec<Option<&[f64]>> = vec![None; n_factors];
    let mut factor_global: Vec<Option<&[u32]>> = vec![None; n_factors];
    for b in blocks {
        if factor_diag[b.q].is_none() {
            factor_diag[b.q] = Some(&b.diag_q);
            factor_global[b.q] = Some(&b.q_global);
        }
        if factor_diag[b.r].is_none() {
            factor_diag[b.r] = Some(&b.diag_r);
            factor_global[b.r] = Some(&b.r_global);
        }
    }

    // Group blocks by factor for fast lookup.
    // first_pairs[f]: blocks where f is the q-factor (entries from C go into f's rows)
    // second_pairs[f]: blocks where f is the r-factor (entries from C^T go into f's rows)
    let mut first_pairs: Vec<Vec<usize>> = vec![Vec::new(); n_factors];
    let mut second_pairs: Vec<Vec<usize>> = vec![Vec::new(); n_factors];
    for (bi, b) in blocks.iter().enumerate() {
        first_pairs[b.q].push(bi);
        second_pairs[b.r].push(bi);
    }

    // --- Pass 1: count NNZ per row ---
    let mut row_nnz = vec![0u32; n_dofs];

    // Diagonals: one entry per active level per factor
    for f_global in factor_global.iter().flatten() {
        for &g in *f_global {
            row_nnz[g as usize] += 1;
        }
    }

    // Off-diagonals from C blocks (placed in q-factor rows)
    for b in blocks {
        for (cj, &g) in b.q_global.iter().enumerate() {
            let nnz_in_row = b.c.indptr[cj + 1] - b.c.indptr[cj];
            row_nnz[g as usize] += nnz_in_row;
        }
        // C^T entries placed in r-factor rows
        for (ck, &g) in b.r_global.iter().enumerate() {
            let nnz_in_row = b.ct.indptr[ck + 1] - b.ct.indptr[ck];
            row_nnz[g as usize] += nnz_in_row;
        }
    }

    // --- Build indptr ---
    let mut indptr = vec![0u32; n_dofs + 1];
    for i in 0..n_dofs {
        indptr[i + 1] = indptr[i] + row_nnz[i];
    }
    let total_nnz = indptr[n_dofs] as usize;

    // --- Pass 2: fill indices and data ---
    let mut indices = vec![0u32; total_nnz];
    let mut data = vec![0.0f64; total_nnz];
    let mut cursor = indptr[..n_dofs].to_vec();

    // Process factor by factor to maintain sorted column order.
    // For each active level of factor f:
    //   1. C^T entries from pairs (p, f) where p < f → columns in p's range (< f's offset)
    //   2. Diagonal entry → column = own global index (in f's range)
    //   3. C entries from pairs (f, r) where r > f → columns in r's range (> f's offset)
    for f in 0..n_factors {
        let f_global = match factor_global[f] {
            Some(g) => g,
            None => continue,
        };
        let f_diag = factor_diag[f].unwrap();
        let n_active_f = f_global.len();

        for compact_j in 0..n_active_f {
            let g = f_global[compact_j] as usize;

            // 1. Entries from pairs (p, f) where p < f — columns in p's range
            for &bi in &second_pairs[f] {
                let b = &blocks[bi];
                // f is the r-factor in this block; row compact_j of C^T
                let start = b.ct.indptr[compact_j] as usize;
                let end = b.ct.indptr[compact_j + 1] as usize;
                for idx in start..end {
                    let compact_p = b.ct.indices[idx] as usize;
                    let pos = cursor[g] as usize;
                    indices[pos] = b.q_global[compact_p];
                    data[pos] = b.ct.data[idx];
                    cursor[g] += 1;
                }
            }

            // 2. Diagonal
            let pos = cursor[g] as usize;
            indices[pos] = g as u32;
            data[pos] = f_diag[compact_j];
            cursor[g] += 1;

            // 3. Entries from pairs (f, r) where r > f — columns in r's range
            for &bi in &first_pairs[f] {
                let b = &blocks[bi];
                // f is the q-factor in this block; row compact_j of C
                let start = b.c.indptr[compact_j] as usize;
                let end = b.c.indptr[compact_j + 1] as usize;
                for idx in start..end {
                    let compact_r = b.c.indices[idx] as usize;
                    let pos = cursor[g] as usize;
                    indices[pos] = b.r_global[compact_r];
                    data[pos] = b.c.data[idx];
                    cursor[g] += 1;
                }
            }
        }
    }

    SparseMatrix::new(indptr, indices, data, n_dofs)
}
