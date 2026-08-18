//! Fused-block correction ladder for collinearity-warned term groups (#281).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::reborrow::ReborrowMut;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
    SymmetricOrdering,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, MatMut, Par, Side};
use schwarz_precond::Operator;

use crate::domain::{Design, Loading};
use crate::error::BuildWarning;
use crate::operator::schwarz::Preconditioner;

/// Factor-nnz budget above which the exact solve is declined (fill is set by graph topology).
const FILL_CAP: usize = 40_000_000;

/// FSAI row-pattern cap: keeps setup O(nnz · cap²) under heavy-tailed level degrees.
const FSAI_ROW_CAP: usize = 48;

/// The assembled (weighted, whitened) Gram of one warned term group, in sparse form.
pub(crate) struct FusedGram {
    /// Global DOF ranges of the fused terms, concatenated in local order.
    pub(crate) spans: Vec<Range<usize>>,
    pub(crate) n_local: usize,
    pub(crate) diag: Vec<f64>,
    /// Strict upper-triangular entries keyed by `(row, col)` with `row < col`.
    pub(crate) off: HashMap<(u32, u32), f64>,
}

pub(crate) fn assemble_gram(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    terms: &[usize],
) -> FusedGram {
    let spans: Vec<Range<usize>> = terms
        .iter()
        .map(|&t| {
            let meta = &design.terms[t];
            meta.offset..meta.offset + meta.columns.len() * meta.n_levels
        })
        .collect();
    let local_bases: Vec<usize> = spans
        .iter()
        .scan(0, |acc, span| {
            let base = *acc;
            *acc += span.len();
            Some(base)
        })
        .collect();
    let n_local = spans.iter().map(|s| s.len()).sum();

    let mut diag = vec![0.0f64; n_local];
    let mut off: HashMap<(u32, u32), f64> = HashMap::new();
    let mut ents: Vec<(usize, f64)> = Vec::new();
    for i in 0..design.n_obs {
        let w = weights.map_or(1.0, |ws| ws[i]);
        ents.clear();
        for (span, &t) in terms.iter().enumerate() {
            let meta = &design.terms[t];
            let level = design.frame.level_column(t)[i] as usize;
            for (c, loading) in meta.columns.iter().enumerate() {
                let coef = match loading {
                    Loading::Constant => 1.0,
                    Loading::Covariate(k) => design.frame.loading_column(*k as usize)[i],
                };
                let local = local_bases[span] + meta.column_base(c) + level - meta.offset;
                ents.push((local, coef));
            }
        }
        for (p, &(dp, cp)) in ents.iter().enumerate() {
            diag[dp] += w * cp * cp;
            for &(dq, cq) in &ents[p + 1..] {
                let key = if dp <= dq {
                    (dp as u32, dq as u32)
                } else {
                    (dq as u32, dp as u32)
                };
                *off.entry(key).or_insert(0.0) += w * cp * cq;
            }
        }
    }

    FusedGram {
        spans,
        n_local,
        diag,
        off,
    }
}

/// Ladder rung chosen by the fill gate: exact where affordable, approximate inverse otherwise.
enum FusedFactor {
    /// Sparse LDLᵀ of the whole fused Gram; cures any topology but pays the fill.
    Exact {
        symbolic: SymbolicCholesky<usize>,
        l_values: Vec<f64>,
    },
    /// `A⁻¹ ≈ S GᵀG S` on the prescaled Gram, pattern capped lower(A); fill-free.
    Fsai {
        scale: Vec<f64>,
        row_ptr: Vec<usize>,
        col_idx: Vec<u32>,
        values: Vec<f64>,
    },
}

fn exact_factor(gram: &FusedGram, fill_cap: usize) -> Option<FusedFactor> {
    let n_local = gram.n_local;
    let mut triplets = Vec::with_capacity(n_local + gram.off.len());
    for (d, &v) in gram.diag.iter().enumerate() {
        triplets.push(Triplet::new(d, d, v));
    }
    for (&(r, c), &v) in &gram.off {
        triplets.push(Triplet::new(r as usize, c as usize, v));
    }
    let a_upper =
        SparseColMat::<usize, f64>::try_new_from_triplets(n_local, n_local, &triplets).ok()?;

    let symbolic = factorize_symbolic_cholesky(
        a_upper.symbolic(),
        Side::Upper,
        SymmetricOrdering::Amd,
        CholeskySymbolicParams::default(),
    )
    .ok()?;
    if symbolic.len_val() > fill_cap {
        return None;
    }

    // Null pivots become LARGE; tiny ones amplify nulls ~1e12x and fake convergence.
    let d_max = gram.diag.iter().copied().fold(0.0, f64::max);
    let signs = vec![1i8; n_local];
    let regularization = LdltRegularization {
        dynamic_regularization_signs: Some(&signs),
        dynamic_regularization_delta: d_max,
        dynamic_regularization_epsilon: 1e-12 * d_max,
    };
    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let mut mem = MemBuffer::new(
        symbolic.factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default()),
    );
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a_upper.as_ref(),
            Side::Upper,
            regularization,
            Par::Seq,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .ok()?;

    Some(FusedFactor::Exact { symbolic, l_values })
}

/// Infallible bottom rung; each row solves a small principal subsystem, so no shift is needed.
fn fsai_factor(gram: &FusedGram) -> FusedFactor {
    let n = gram.n_local;
    let scale: Vec<f64> = gram
        .diag
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
        .collect();
    let mut entries: HashMap<(u32, u32), f64> = HashMap::with_capacity(gram.off.len());
    let mut lower_adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for (&(r, c), &v) in &gram.off {
        let sv = v * scale[r as usize] * scale[c as usize];
        entries.insert((r.min(c), r.max(c)), sv);
        lower_adj[r.max(c) as usize].push((r.min(c), sv));
    }
    let unit_diag: Vec<f64> = gram
        .diag
        .iter()
        .map(|&d| if d > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<u32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);
    let mut small = vec![0.0f64; (FSAI_ROW_CAP + 1) * (FSAI_ROW_CAP + 1)];
    let mut rhs = vec![0.0f64; FSAI_ROW_CAP + 1];
    let mut idx: Vec<u32> = Vec::with_capacity(FSAI_ROW_CAP + 1);
    for (i, adj) in lower_adj.iter_mut().enumerate() {
        if adj.len() > FSAI_ROW_CAP {
            adj.sort_unstable_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            adj.truncate(FSAI_ROW_CAP);
        }
        idx.clear();
        idx.extend(adj.iter().map(|&(c, _)| c));
        idx.sort_unstable();
        idx.push(i as u32);
        let m = idx.len();
        for a in 0..m {
            for b in 0..=a {
                let (r, c) = (idx[b].min(idx[a]), idx[b].max(idx[a]));
                let v = if r == c {
                    unit_diag[r as usize] + 1e-10
                } else {
                    entries.get(&(r, c)).copied().unwrap_or(0.0)
                };
                small[a * m + b] = v;
                small[b * m + a] = v;
            }
        }
        rhs[..m].fill(0.0);
        rhs[m - 1] = 1.0;
        dense_spd_solve(&mut small[..m * m], m, &mut rhs[..m]);
        let inv_sqrt = 1.0 / rhs[m - 1].max(1e-30).sqrt();
        for (&col, &v) in idx.iter().zip(&rhs[..m]) {
            let g = v * inv_sqrt;
            if g != 0.0 {
                col_idx.push(col);
                values.push(g);
            }
        }
        row_ptr.push(values.len());
    }

    FusedFactor::Fsai {
        scale,
        row_ptr,
        col_idx,
        values,
    }
}

/// In-place Cholesky solve; near-null pivots get a jitter floor instead of failing.
fn dense_spd_solve(a: &mut [f64], m: usize, rhs: &mut [f64]) {
    for j in 0..m {
        let mut d = a[j * m + j];
        for k in 0..j {
            d -= a[j * m + k] * a[j * m + k];
        }
        if d <= 1e-14 {
            d = 1e-12;
        }
        let sd = d.sqrt();
        a[j * m + j] = sd;
        for i in j + 1..m {
            let mut v = a[i * m + j];
            for k in 0..j {
                v -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = v / sd;
        }
    }
    for i in 0..m {
        let mut v = rhs[i];
        for k in 0..i {
            v -= a[i * m + k] * rhs[k];
        }
        rhs[i] = v / a[i * m + i];
    }
    for i in (0..m).rev() {
        let mut v = rhs[i];
        for k in i + 1..m {
            v -= a[k * m + i] * rhs[k];
        }
        rhs[i] = v / a[i * m + i];
    }
}

/// Local solve over one warned term group, applied additively on top of the base
/// preconditioner and never serialized (see docs part 3, §5).
pub(crate) struct FusedBlockSolve {
    factor: FusedFactor,
    /// Global DOF ranges of the fused terms, concatenated in local order.
    spans: Vec<Range<usize>>,
    n_local: usize,
}

impl FusedBlockSolve {
    /// One solve per connected component of the warned term pairs;
    /// must run after whitening (the Gram is read from the reparametrized frame).
    pub(crate) fn build_all(
        design: &Design<'_>,
        weights: Option<&[f64]>,
        warnings: &[BuildWarning],
    ) -> Vec<Self> {
        let n_terms = design.terms.len();
        let mut parent: Vec<usize> = (0..n_terms).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let mut warned = vec![false; n_terms];
        for w in warnings {
            if let BuildWarning::CollinearSlopeCovariate { slope, term, .. } = w {
                warned[slope.term] = true;
                warned[*term] = true;
                let (a, b) = (root(&mut parent, slope.term), root(&mut parent, *term));
                parent[a] = b;
            }
        }
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for t in (0..n_terms).filter(|&t| warned[t]) {
            components.entry(root(&mut parent, t)).or_default().push(t);
        }
        let mut groups: Vec<Vec<usize>> = components.into_values().collect();
        groups.sort();
        groups
            .into_iter()
            .map(|terms| Self::build(design, weights, &terms, FILL_CAP))
            .collect()
    }

    fn build(
        design: &Design<'_>,
        weights: Option<&[f64]>,
        terms: &[usize],
        fill_cap: usize,
    ) -> Self {
        let gram = assemble_gram(design, weights, terms);
        let factor = exact_factor(&gram, fill_cap).unwrap_or_else(|| fsai_factor(&gram));
        Self {
            factor,
            spans: gram.spans,
            n_local: gram.n_local,
        }
    }

    /// `y[spans] += A_fused⁻¹ x[spans]` (approximately on the FSAI rung).
    fn solve_add(&self, x: &[f64], y: &mut [f64], scratch: &mut FusedScratch) {
        let local = &mut scratch.local[..self.n_local];
        let mut base = 0;
        for span in &self.spans {
            local[base..base + span.len()].copy_from_slice(&x[span.clone()]);
            base += span.len();
        }
        match &self.factor {
            FusedFactor::Exact { symbolic, l_values } => {
                let ldlt = LdltRef::new(symbolic, l_values);
                let mut rhs = MatMut::from_column_major_slice_mut(local, self.n_local, 1);
                ldlt.solve_in_place_with_conj(
                    Conj::No,
                    rhs.rb_mut(),
                    Par::Seq,
                    MemStack::new(&mut scratch.mem),
                );
            }
            FusedFactor::Fsai {
                scale,
                row_ptr,
                col_idx,
                values,
            } => {
                let work = &mut scratch.work[..self.n_local];
                for (l, &s) in local.iter_mut().zip(scale) {
                    *l *= s;
                }
                for (i, wi) in work.iter_mut().enumerate() {
                    let mut z = 0.0;
                    for p in row_ptr[i]..row_ptr[i + 1] {
                        z += values[p] * local[col_idx[p] as usize];
                    }
                    *wi = z;
                }
                local.fill(0.0);
                for (i, &zi) in work.iter().enumerate() {
                    for p in row_ptr[i]..row_ptr[i + 1] {
                        local[col_idx[p] as usize] += values[p] * zi;
                    }
                }
                for (l, &s) in local.iter_mut().zip(scale) {
                    *l *= s;
                }
            }
        }
        let mut base = 0;
        for span in &self.spans {
            for (yi, &li) in y[span.clone()].iter_mut().zip(&local[base..]) {
                *yi += li;
            }
            base += span.len();
        }
    }
}

/// Reused per-apply buffers: the solve sits in the LSMR hot loop.
struct FusedScratch {
    local: Vec<f64>,
    work: Vec<f64>,
    mem: MemBuffer,
}

/// The base preconditioner plus the additive fused-block corrections.
pub(crate) struct FusedPreconditioner<'a> {
    base: &'a Preconditioner,
    blocks: &'a [FusedBlockSolve],
    scratch: Mutex<FusedScratch>,
}

impl<'a> FusedPreconditioner<'a> {
    pub(crate) fn new(base: &'a Preconditioner, blocks: &'a [FusedBlockSolve]) -> Self {
        let n_max = blocks.iter().map(|b| b.n_local).max().unwrap_or(0);
        let req = blocks
            .iter()
            .filter_map(|b| match &b.factor {
                FusedFactor::Exact { symbolic, .. } => {
                    Some(symbolic.solve_in_place_scratch::<f64>(1, Par::Seq))
                }
                FusedFactor::Fsai { .. } => None,
            })
            .fold(StackReq::empty(), StackReq::or);
        Self {
            base,
            blocks,
            scratch: Mutex::new(FusedScratch {
                local: vec![0.0; n_max],
                work: vec![0.0; n_max],
                mem: MemBuffer::new(req),
            }),
        }
    }

    fn solve_add_all(&self, x: &[f64], y: &mut [f64]) {
        let mut scratch = self.scratch.lock().unwrap();
        for block in self.blocks {
            block.solve_add(x, y, &mut scratch);
        }
    }
}

impl Operator for FusedPreconditioner<'_> {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }

    fn ncols(&self) -> usize {
        self.base.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.base.apply(x, y)?;
        self.solve_add_all(x, y);
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        <Preconditioner as Operator>::apply_adjoint(self.base, x, y)?;
        self.solve_add_all(x, y);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl FusedBlockSolve {
        pub(crate) fn build_for_test(
            design: &Design<'_>,
            terms: &[usize],
            fill_cap: usize,
        ) -> Self {
            Self::build(design, None, terms, fill_cap)
        }

        pub(crate) fn is_exact_for_test(&self) -> bool {
            matches!(self.factor, FusedFactor::Exact { .. })
        }
    }
}
