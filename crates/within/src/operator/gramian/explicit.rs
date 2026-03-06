use schwarz_precond::{Operator, SparseMatrix};

use super::accumulator::PairAccumulator;
use super::csr_assembly::{build_symmetric_csr, CompactIndexMaps};
use super::Gramian;
use crate::domain::{PairBlockData, WeightedDesign};
use crate::observation::{FactorMeta, ObservationStore};

impl Gramian {
    pub fn build<S: ObservationStore>(design: &WeightedDesign<S>) -> Self {
        Self {
            matrix: build_full_matrix(design),
        }
    }

    /// Build a Gramian containing only the single factor pair `(q, r)`.
    pub fn build_for_pair<S: ObservationStore>(
        design: &WeightedDesign<S>,
        q: usize,
        r: usize,
    ) -> Self {
        Self {
            matrix: build_pair_matrix(design, q, r),
        }
    }

    /// Build a Gramian scoped to a single connected component in compact local
    /// index space.
    pub fn build_for_component<S: ObservationStore>(
        design: &WeightedDesign<S>,
        q: usize,
        r: usize,
        component_global_indices: &[u32],
    ) -> Self {
        Self {
            matrix: build_component_matrix(design, q, r, component_global_indices),
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
            matrix: compose_gramian_from_blocks(blocks, factors, n_dofs),
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
    let n_unique = design.store.n_unique();
    let n_factors = design.n_factors();

    let mut diag_counts: Vec<Vec<f64>> = design
        .factors
        .iter()
        .map(|f| vec![0.0; f.n_levels])
        .collect();

    let n_pairs = n_factors * (n_factors - 1) / 2;
    let mut pair_info: Vec<(usize, usize)> = Vec::with_capacity(n_pairs);
    let mut pair_tables: Vec<PairAccumulator> = Vec::with_capacity(n_pairs);

    for q in 0..n_factors {
        for r in (q + 1)..n_factors {
            let fq = &design.factors[q];
            let fr = &design.factors[r];
            pair_info.push((q, r));
            pair_tables.push(PairAccumulator::new(fq.n_levels, fr.n_levels, n_unique));
        }
    }

    for uid in 0..n_unique {
        let w = design.uid_weight(uid);
        for (q, diag_q) in diag_counts.iter_mut().enumerate() {
            let j = design.store.unique_level(uid, q) as usize;
            diag_q[j] += w;
        }

        for (pi, &(q, r)) in pair_info.iter().enumerate() {
            let j = design.store.unique_level(uid, q) as usize;
            let k = design.store.unique_level(uid, r) as usize;
            pair_tables[pi].add(j, k, w);
        }
    }

    build_symmetric_csr(n_dofs, |emit| {
        for (q, counts) in diag_counts.iter().enumerate() {
            let fq = &design.factors[q];
            for (j, &cnt) in counts.iter().enumerate() {
                if cnt > 0.0 {
                    let row = fq.offset + j;
                    emit(row, row, cnt);
                }
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
    let n_unique = design.store.n_unique();
    let fq = &design.factors[q];
    let fr = &design.factors[r];

    let mut diag_q = vec![0.0; fq.n_levels];
    let mut diag_r = vec![0.0; fr.n_levels];
    let mut table = PairAccumulator::new(fq.n_levels, fr.n_levels, n_unique);

    for uid in 0..n_unique {
        let w = design.uid_weight(uid);
        let j = design.store.unique_level(uid, q) as usize;
        let k = design.store.unique_level(uid, r) as usize;
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

fn build_component_matrix<S: ObservationStore>(
    design: &WeightedDesign<S>,
    q: usize,
    r: usize,
    component_global_indices: &[u32],
) -> SparseMatrix {
    let n_unique = design.store.n_unique();
    let maps = CompactIndexMaps::build(&design.factors, q, r, component_global_indices);
    let n_local = maps.n_local();

    let mut diag_q = vec![0.0; maps.n_active_q];
    let mut diag_r = vec![0.0; maps.n_active_r];
    let mut table = PairAccumulator::new(maps.n_active_q, maps.n_active_r, n_unique);

    for uid in 0..n_unique {
        let j = design.store.unique_level(uid, q) as usize;
        let k = design.store.unique_level(uid, r) as usize;
        let cj = maps.q_compact[j];
        let ck = maps.r_compact[k];
        if cj == u32::MAX || ck == u32::MAX {
            continue;
        }

        let w = design.uid_weight(uid);
        let cj = cj as usize;
        let ck = ck as usize;
        diag_q[cj] += w;
        diag_r[ck] += w;
        table.add(cj, ck, w);
    }

    let first_block_size = maps.n_active_q;
    build_symmetric_csr(n_local, |emit| {
        for (j, &cnt) in diag_q.iter().enumerate() {
            if cnt > 0.0 {
                emit(j, j, cnt);
            }
        }
        for (k, &cnt) in diag_r.iter().enumerate() {
            if cnt > 0.0 {
                let row = first_block_size + k;
                emit(row, row, cnt);
            }
        }

        table.for_each_nonzero(|j, k, cnt| {
            let gj = j;
            let gk = first_block_size + k;
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
