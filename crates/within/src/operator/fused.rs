//! Exact fused-block correction for collinearity-warned term groups (#281).

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

/// Exact sparse LDLᵀ of the Gram over one warned term group, applied additively
/// on top of the base preconditioner and never serialized (see docs part 3, §5).
pub(crate) struct FusedBlockSolve {
    symbolic: SymbolicCholesky<usize>,
    l_values: Vec<f64>,
    /// Global DOF ranges of the fused terms, concatenated in local order.
    spans: Vec<Range<usize>>,
    n_local: usize,
}

impl FusedBlockSolve {
    /// One fill-gated solve per connected component of the warned term pairs;
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
            .filter_map(|terms| Self::build(design, weights, &terms))
            .collect()
    }

    fn build(design: &Design<'_>, weights: Option<&[f64]>, terms: &[usize]) -> Option<Self> {
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

        let mut triplets = Vec::with_capacity(n_local + off.len());
        for (d, &v) in diag.iter().enumerate() {
            triplets.push(Triplet::new(d, d, v));
        }
        for (&(r, c), &v) in &off {
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
        if symbolic.len_val() > FILL_CAP {
            return None;
        }

        // Null pivots become LARGE; tiny ones amplify nulls ~1e12x and fake convergence.
        let d_max = diag.iter().copied().fold(0.0, f64::max);
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

        Some(Self {
            symbolic,
            l_values,
            spans,
            n_local,
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
        let ldlt = LdltRef::new(&self.symbolic, &self.l_values);
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
            .map(|b| b.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq))
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
