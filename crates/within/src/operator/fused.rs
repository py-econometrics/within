//! Fused-block correction for collinearity-warned term groups (#281).

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
use faer::sparse::{Pair, SparseColMat, SymbolicSparseColMat, Triplet};
use faer::{Conj, MatMut, Par, Side};
use schwarz_precond::Operator;

use crate::domain::Design;
use crate::error::BuildWarning;
use crate::operator::schwarz::Preconditioner;

/// The assembled (weighted, whitened) Gram of one warned term group, in sparse form.
struct FusedGram {
    /// Global DOF ranges of the fused terms, concatenated in local order.
    spans: Vec<Range<usize>>,
    n_local: usize,
    diag: Vec<f64>,
    /// Strict upper-triangular entries keyed by `(row, col)` with `row < col`.
    off: HashMap<(u32, u32), f64>,
}

fn assemble_gram(design: &Design<'_>, weights: Option<&[f64]>, terms: &[usize]) -> FusedGram {
    let spans: Vec<Range<usize>> = terms
        .iter()
        .map(|&t| {
            let meta = &design.terms[t];
            meta.offset..meta.offset + meta.n_dofs()
        })
        .collect();
    let n_local = spans.iter().map(|s| s.len()).sum();

    // Per term: local base, n_levels, level codes, per-column loadings (None = intercept).
    let mut base = 0;
    let cols: Vec<_> = terms
        .iter()
        .zip(&spans)
        .map(|(&t, span)| {
            let meta = &design.terms[t];
            let loadings: Vec<Option<&[f64]>> = meta
                .columns
                .iter()
                .map(|l| {
                    l.covariate()
                        .map(|&k| design.frame.loading_column(k as usize))
                })
                .collect();
            let entry = (base, meta.n_levels, design.frame.level_column(t), loadings);
            base += span.len();
            entry
        })
        .collect();

    let mut diag = vec![0.0f64; n_local];
    let mut off: HashMap<(u32, u32), f64> = HashMap::new();
    let mut ents: Vec<(usize, f64)> = Vec::new();
    for i in 0..design.n_obs {
        let w = weights.map_or(1.0, |ws| ws[i]);
        ents.clear();
        for (base, n_levels, levels, loadings) in &cols {
            let level = levels[i] as usize;
            for (c, loading) in loadings.iter().enumerate() {
                let coef = loading.map_or(1.0, |s| s[i]);
                ents.push((base + c * n_levels + level, coef));
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

/// Sparse LDLᵀ of one fused Gram, under a fill-reducing ordering.
struct FusedFactor {
    symbolic: Box<SymbolicCholesky<usize>>,
    l_values: Vec<f64>,
}

fn exact_factor(gram: &FusedGram, budget: usize) -> Option<FusedFactor> {
    let n_local = gram.n_local;
    // Gate on the pattern alone; numeric staging is paid only once accepted.
    let mut pairs = Vec::with_capacity(n_local + gram.off.len());
    for d in 0..n_local {
        pairs.push(Pair { row: d, col: d });
    }
    for &(r, c) in gram.off.keys() {
        pairs.push(Pair {
            row: r as usize,
            col: c as usize,
        });
    }
    let (pattern, _) =
        SymbolicSparseColMat::<usize>::try_new_from_indices(n_local, n_local, &pairs).ok()?;
    drop(pairs);
    let symbolic = factorize_symbolic_cholesky(
        pattern.as_ref(),
        Side::Upper,
        SymmetricOrdering::Amd,
        CholeskySymbolicParams::default(),
    )
    .ok()?;
    let factor_bytes = size_of::<f64>() + size_of::<usize>();
    if symbolic.len_val().saturating_mul(factor_bytes) > budget {
        return None;
    }

    let mut triplets = Vec::with_capacity(n_local + gram.off.len());
    for (d, &v) in gram.diag.iter().enumerate() {
        triplets.push(Triplet::new(d, d, v));
    }
    for (&(r, c), &v) in &gram.off {
        triplets.push(Triplet::new(r as usize, c as usize, v));
    }
    let a_upper =
        SparseColMat::<usize, f64>::try_new_from_triplets(n_local, n_local, &triplets).ok()?;

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

    Some(FusedFactor {
        symbolic: Box::new(symbolic),
        l_values,
    })
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
        budget: usize,
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
        let mut components: Vec<Vec<usize>> = vec![Vec::new(); n_terms];
        for t in (0..n_terms).filter(|&t| warned[t]) {
            components[root(&mut parent, t)].push(t);
        }
        components.retain(|g| !g.is_empty());
        components
            .into_iter()
            .filter_map(|terms| Self::build(design, weights, &terms, budget))
            .collect()
    }

    /// `None` where the symbolic factor exceeds the budget; the group then goes uncorrected.
    fn build(
        design: &Design<'_>,
        weights: Option<&[f64]>,
        terms: &[usize],
        budget: usize,
    ) -> Option<Self> {
        let gram = assemble_gram(design, weights, terms);
        Some(Self {
            factor: exact_factor(&gram, budget)?,
            spans: gram.spans,
            n_local: gram.n_local,
        })
    }

    /// `y[spans] += A_fused⁻¹ x[spans]`.
    fn solve_add(&self, x: &[f64], y: &mut [f64], scratch: &mut FusedScratch) {
        let local = &mut scratch.local[..self.n_local];
        let mut base = 0;
        for span in &self.spans {
            local[base..base + span.len()].copy_from_slice(&x[span.clone()]);
            base += span.len();
        }
        let ldlt = LdltRef::new(&self.factor.symbolic, &self.factor.l_values);
        let mut rhs = MatMut::from_column_major_slice_mut(local, self.n_local, 1);
        ldlt.solve_in_place_with_conj(
            Conj::No,
            rhs.rb_mut(),
            Par::Seq,
            MemStack::new(&mut scratch.mem),
        );
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
            .map(|b| b.factor.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq))
            .fold(StackReq::empty(), StackReq::or);
        Self {
            base,
            blocks,
            scratch: Mutex::new(FusedScratch {
                local: vec![0.0; n_max],
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
            budget: usize,
        ) -> Option<Self> {
            Self::build(design, None, terms, budget)
        }
    }
}
