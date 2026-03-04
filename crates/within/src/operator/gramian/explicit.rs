use schwarz_precond::{Operator, SparseMatrix};

use super::accumulator::PairAccumulator;
use super::csr_assembly::{build_symmetric_csr, CompactIndexMaps};
use super::Gramian;
use crate::domain::WeightedDesign;
use crate::observation::ObservationStore;

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
