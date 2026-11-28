//! PROTOTYPE(#607): the rigid coarse space of `feature/reduced-solve` (3ba7a0cd), ported onto the
//! reduced system: `Z`, `SZ` and the factor of `E = ZᵀSZ`, for the two-level map
//! `PᵀBP + Q` with `Q = Z E⁻¹ Zᵀ`, `P = I − SQ`.

mod assembled;
mod rigid;
pub(crate) mod spaces;

use std::ops::Range;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::perm::PermRef;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
    SymmetricOrdering,
};
use faer::sparse::{SparseColMat, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};
use rayon::prelude::*;

use super::Gramian;

/// Entries from which rayon beats one thread: the gather `Mᵀx` pays later than the row-blocked
/// `M c`.
const PARALLEL_T_MUL: usize = 1 << 18;
const PARALLEL_MUL_ADD: usize = 1 << 16;

/// A coarse direction keeping less than this share of its energy is null: a column of `Z` once
/// the eliminated term is fitted, or a pivot of `E` at unit diagonal.
const NULL_ENERGY_SHARE: f64 = 1e-10;

/// Why the coarse build stays one-level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoarseDecline {
    /// Gluing leaves no coarse column worth one, or the design has no anchor term.
    NoCoarseSpace,
    /// The factor of `E` would hold more values than there are observations.
    OverBudget,
}

impl Gramian<'_> {
    /// PROTOTYPE(#607): the rigid coarse solve over the unknowns, or why there is none.
    ///
    /// PROTOTYPE(#612): `PROBE_SPACE` picks another coarse space; see [`spaces::Space`].
    pub(crate) fn rigid_coarse(&self) -> Result<CoarseSolve, CoarseDecline> {
        spaces::build(self.design, spaces::space())
    }
}

/// The coarse space `Z`, `SZ`, and the factor of `E = ZᵀSZ`.
pub(crate) struct CoarseSolve {
    z: TwoWay,
    /// `SZ`, less `hubs` where there are wide levels.
    sz: TwoWay,
    /// Wide levels' rows: the factor is of `[G s; sᵀ E₀]`, whose Schur complement is `E`.
    hubs: Option<HubCorrection>,
    factor: CoarseFactor,
}

impl std::fmt::Debug for CoarseSolve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoarseSolve")
            .field("columns", &self.z.ncols())
            .field("factor_values", &self.factor.l_values.len())
            .finish()
    }
}

impl CoarseSolve {
    /// PROTOTYPE(#612): the coarse solve over dense columns `w`, `SW` given; declines when `E`
    /// has no live pivot.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn from_dense(
        n: usize,
        w: &[Vec<f64>],
        sw: &[Vec<f64>],
    ) -> Result<Self, CoarseDecline> {
        let k = w.len();
        let dense = |cols: &[Vec<f64>]| {
            let parts = vec![(cols.iter().enumerate())
                .flat_map(|(c, col)| {
                    (col.iter().enumerate())
                        .filter(|e| *e.1 != 0.0)
                        .map(move |(r, &v)| (c as u32, r as u32, v))
                })
                .collect::<Vec<_>>()];
            TwoWay::from_parts(n, k, parts)
        };
        let mut lower = Vec::new();
        for c in 0..k {
            for i in c..k {
                let v: f64 = w[i].iter().zip(&sw[c]).map(|(a, b)| a * b).sum();
                lower.push((c as u32, i as u32, v));
            }
        }
        let factor = CoarseFactor::new(k, lower, None, usize::MAX)?;
        Ok(Self {
            z: dense(w),
            sz: dense(sw),
            hubs: None,
            factor,
        })
    }

    /// Number of coarse columns.
    pub(crate) fn n_columns(&self) -> usize {
        self.z.ncols()
    }

    /// Hub rows bordering `E`.
    pub(crate) fn n_hubs(&self) -> usize {
        self.hubs.as_ref().map_or(0, |h| h.rows.rank())
    }

    /// Values in the factor of `E`.
    pub(crate) fn factor_values(&self) -> usize {
        self.factor.l_values.len()
    }

    /// Stored nonzeros of `Z`, `SZ` and `F`.
    pub(crate) fn nnz(&self) -> usize {
        self.z.nnz() + self.sz.nnz() + self.hubs.as_ref().map_or(0, |h| h.f.nnz())
    }

    /// PROTOTYPE(#612): `Z`'s columns as `(unknown, value)` lists.
    pub(super) fn z_columns(&self) -> Vec<Vec<(u32, f64)>> {
        let z = &self.z;
        (0..z.ncols())
            .map(|c| {
                (z.col_ptr[c]..z.col_ptr[c + 1])
                    .map(|k| (z.col_idx[k], z.col_val[k]))
                    .collect()
            })
            .collect()
    }

    /// `out = Zᵀx`.
    pub(crate) fn z_t_mul(&self, x: &[f64], out: &mut [f64]) {
        self.z.t_mul(x, out);
    }

    /// `out += Z c`.
    pub(crate) fn add_z(&self, c: &[f64], out: &mut [f64]) {
        self.z.mul_add(c, 1.0, out);
    }

    /// `v ← E⁻¹v`.
    pub(crate) fn solve(&self, v: &mut [f64]) {
        self.factor.solve(v);
    }

    /// `out −= SZ c`.
    pub(crate) fn sub_sz(&self, c: &[f64], out: &mut [f64]) {
        self.sz.mul_add(c, -1.0, out);
        if let Some(hubs) = &self.hubs {
            hubs.add(c, out);
        }
    }

    /// `out = SZᵀ y`.
    pub(crate) fn sz_t_mul(&self, y: &[f64], out: &mut [f64]) {
        self.sz.t_mul(y, out);
        if let Some(hubs) = &self.hubs {
            hubs.sub_transpose(y, out);
        }
    }
}

/// The LDLᵀ of `D E D`, `D` scaling `E` to a unit diagonal.
struct CoarseFactor {
    symbolic: SymbolicCholesky<u32>,
    l_values: Vec<f64>,
    scale: Vec<f64>,
}

/// `(column, row, value)` triplets by column then row, duplicates summed in input order, zero
/// sums dropped.
fn merged(mut triplets: Vec<(u32, u32, f64)>) -> Vec<(u32, u32, f64)> {
    triplets.par_sort_by_key(|e| (e.0, e.1));
    (triplets.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)))
        .filter_map(|run| {
            let v = run.iter().fold(0.0, |v, e| v + e.2);
            (v != 0.0).then_some((run[0].0, run[0].1, v))
        })
        .collect()
}

/// Start offsets of `n_keys` runs from the key of each item.
fn offsets(n_keys: usize, keys: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut offsets = vec![0usize; n_keys + 1];
    for k in keys {
        offsets[k + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    offsets
}

/// `0..n` cut into consecutive ranges whose `size`s sum to at least `min`, the last maybe less.
fn cut(n: usize, min: usize, size: impl Fn(usize) -> usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let (mut first, mut total) = (0, 0);
    for k in 0..n {
        total += size(k);
        if total >= min {
            ranges.push(first..k + 1);
            (first, total) = (k + 1, 0);
        }
    }
    if first < n {
        ranges.push(first..n);
    }
    ranges
}

impl CoarseFactor {
    /// The LDLᵀ of `E`, lower triplets `lower` over its `coarse` columns bordered by the `hubs`'
    /// rows, at unit diagonal, the hubs eliminated last; declines past `budget` factor values.
    fn new(
        coarse: usize,
        mut lower: Vec<(u32, u32, f64)>,
        hubs: Option<&HubRows>,
        budget: usize,
    ) -> Result<Self, CoarseDecline> {
        let dim = coarse + hubs.map_or(0, HubRows::rank);
        lower.extend(hubs.into_iter().flat_map(|h| h.lower(coarse)));
        let lower = merged(lower);
        let mut diagonal = vec![0.0; dim];
        for &(c, r, v) in &lower {
            if c == r {
                diagonal[c as usize] = v;
            }
        }
        // Unit diagonal: null pivots then sit at rounding level, far below any live mode.
        let scale: Vec<f64> = (diagonal.iter())
            .map(|&d| if d > 0.0 { d.sqrt().recip() } else { 1.0 })
            .collect();
        let lower: Vec<(u32, u32, f64)> = (lower.into_iter())
            .map(|(c, r, v)| (c, r, v * (scale[c as usize] * scale[r as usize])))
            .collect();
        let d_max = (lower.iter())
            .filter(|e| e.0 == e.1)
            .map(|e| e.2)
            .fold(0.0, f64::max);
        let col_ptr = (offsets(dim, lower.iter().map(|e| e.0 as usize)).into_iter())
            .map(|k| k as u32)
            .collect();
        let (row_idx, values) = lower.into_iter().map(|(_, r, v)| (r, v)).unzip();
        let pattern = SymbolicSparseColMat::new_checked(dim, dim, col_ptr, None, row_idx);
        let matrix = SparseColMat::new(pattern, values);
        let (fwd, inv) = hubs_last(matrix.symbolic(), coarse)?;
        let symbolic = factorize_symbolic_cholesky(
            matrix.symbolic(),
            Side::Lower,
            SymmetricOrdering::Custom(PermRef::new_checked(&fwd, &inv, dim)),
            CholeskySymbolicParams::default(),
        )
        .map_err(|_| CoarseDecline::NoCoarseSpace)?;
        if symbolic.len_val() > budget {
            return Err(CoarseDecline::OverBudget);
        }
        // Null pivots become LARGE, as in the fused blocks: tiny ones amplify nulls.
        let signs = vec![1i8; dim];
        let regularization = LdltRegularization {
            dynamic_regularization_signs: Some(&signs),
            dynamic_regularization_delta: d_max,
            dynamic_regularization_epsilon: NULL_ENERGY_SHARE * d_max,
        };
        let mut l_values = vec![0.0; symbolic.len_val()];
        let mut mem = MemBuffer::new(
            symbolic.factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default()),
        );
        symbolic
            .factorize_numeric_ldlt(
                &mut l_values,
                matrix.as_ref(),
                Side::Lower,
                regularization,
                Par::Seq,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .map_err(|_| CoarseDecline::NoCoarseSpace)?;
        if !l_values.iter().all(|v| v.is_finite()) {
            return Err(CoarseDecline::NoCoarseSpace);
        }
        Ok(Self {
            symbolic,
            l_values,
            scale,
        })
    }

    /// `E⁻¹v`: the wide levels' rows ride along with a zero right-hand side.
    fn solve(&self, v: &mut [f64]) {
        let dim = self.scale.len();
        let mut x = v.to_vec();
        x.resize(dim, 0.0);
        x.iter_mut().zip(&self.scale).for_each(|(x, s)| *x *= s);
        let mut buf = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
        LdltRef::new(&self.symbolic, &self.l_values).solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(&mut x, dim, 1),
            Par::Seq,
            MemStack::new(&mut buf),
        );
        x.iter_mut().zip(&self.scale).for_each(|(x, s)| *x *= s);
        let m = v.len();
        v.copy_from_slice(&x[..m]);
    }
}

/// AMD over the first `coarse` rows, the rest after them. Eliminated last, a hub's coupling rows
/// take its cancellation into their own pivots, where null modes surface at rounding; eliminated
/// first, they leave it in the coarse block (Björck 1996, ch. 6, block angular problems).
fn hubs_last(
    pattern: SymbolicSparseColMatRef<'_, u32>,
    coarse: usize,
) -> Result<(Vec<u32>, Vec<u32>), CoarseDecline> {
    let dim = pattern.nrows();
    let (mut block_ptr, mut block_rows) = (vec![0u32], Vec::new());
    for c in 0..coarse {
        let rows = pattern.row_idx_of_col_raw(c).iter();
        block_rows.extend(rows.take_while(|&&r| (r as usize) < coarse));
        block_ptr.push(block_rows.len() as u32);
    }
    // AMD orders the pattern of A + Aᵀ: the lower block serves as is.
    let block = SymbolicSparseColMat::new_checked(coarse, coarse, block_ptr, None, block_rows);
    let (mut fwd, mut inv) = (vec![0u32; coarse], vec![0u32; coarse]);
    let mut mem = MemBuffer::new(amd::order_scratch::<u32>(coarse, block.compute_nnz()));
    amd::order(
        &mut fwd,
        &mut inv,
        block.as_ref(),
        amd::Control::default(),
        MemStack::new(&mut mem),
    )
    .map_err(|_| CoarseDecline::NoCoarseSpace)?;
    fwd.extend(coarse as u32..dim as u32);
    inv.extend(coarse as u32..dim as u32);
    Ok((fwd, inv))
}

/// Sparse `n × m` held both ways: by column for `Mᵀx`, by nonempty row for `M c`.
struct TwoWay {
    col_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    col_val: Vec<f64>,
    rows: Vec<u32>,
    row_ptr: Vec<usize>,
    row_idx: Vec<u32>,
    row_val: Vec<f64>,
}

impl TwoWay {
    /// From `(column, row, value)` parts in column order.
    fn from_parts(n: usize, m: usize, parts: Vec<Vec<(u32, u32, f64)>>) -> Self {
        let total: usize = parts.iter().map(Vec::len).sum();
        let col_ptr = offsets(m, parts.iter().flatten().map(|e| e.0 as usize));
        let (mut col_idx, mut col_val) = (Vec::with_capacity(total), Vec::with_capacity(total));
        for (_, r, v) in parts.into_iter().flatten() {
            col_idx.push(r);
            col_val.push(v);
        }
        Self::from_columns(n, col_ptr, col_idx, col_val)
    }

    /// Adds the row view to the columns.
    fn from_columns(n: usize, col_ptr: Vec<usize>, col_idx: Vec<u32>, col_val: Vec<f64>) -> Self {
        let total = col_idx.len();
        let row_ptr = offsets(n, col_idx.iter().map(|&r| r as usize));
        let (mut row_idx, mut row_val) = (vec![0u32; total], vec![0.0; total]);
        let mut cursor = row_ptr.clone();
        for c in 0..col_ptr.len() - 1 {
            for k in col_ptr[c]..col_ptr[c + 1] {
                let slot = &mut cursor[col_idx[k] as usize];
                row_idx[*slot] = c as u32;
                row_val[*slot] = col_val[k];
                *slot += 1;
            }
        }
        let rows: Vec<u32> = (0..n as u32)
            .filter(|&r| row_ptr[r as usize + 1] > row_ptr[r as usize])
            .collect();
        let row_ptr = (rows.iter().map(|&r| row_ptr[r as usize]))
            .chain(row_ptr.last().copied())
            .collect();
        Self {
            col_ptr,
            col_idx,
            col_val,
            rows,
            row_ptr,
            row_idx,
            row_val,
        }
    }

    fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// PROTOTYPE(#612): these columns followed by `extra`, each `(row, value)` ascending.
    fn with_columns(self, n: usize, extra: impl Iterator<Item = Vec<(u32, f64)>>) -> Self {
        let (mut col_ptr, mut col_idx, mut col_val) = (self.col_ptr, self.col_idx, self.col_val);
        for col in extra {
            for (r, v) in col {
                col_idx.push(r);
                col_val.push(v);
            }
            col_ptr.push(col_idx.len());
        }
        Self::from_columns(n, col_ptr, col_idx, col_val)
    }

    /// PROTOTYPE(#612): row `r`'s entry span in the row view, if it has any.
    fn row_span(&self, r: u32) -> Option<(usize, usize)> {
        let i = self.rows.binary_search(&r).ok()?;
        Some((self.row_ptr[i], self.row_ptr[i + 1]))
    }

    fn ncols(&self) -> usize {
        self.col_ptr.len() - 1
    }

    /// `out = Mᵀx`.
    fn t_mul(&self, x: &[f64], out: &mut [f64]) {
        let column = |(c, o): (usize, &mut f64)| {
            let span = self.col_ptr[c]..self.col_ptr[c + 1];
            *o = (self.col_idx[span.clone()].iter().zip(&self.col_val[span]))
                .map(|(&r, &v)| v * x[r as usize])
                .sum();
        };
        if self.nnz() >= PARALLEL_T_MUL {
            out.par_iter_mut().enumerate().for_each(column);
        } else {
            out.iter_mut().enumerate().for_each(column);
        }
    }

    /// `out += alpha M c`.
    fn mul_add(&self, c: &[f64], alpha: f64, out: &mut [f64]) {
        const BLOCK: usize = 4096;
        let row = |i: usize| -> f64 {
            let span = self.row_ptr[i]..self.row_ptr[i + 1];
            (self.row_idx[span.clone()].iter().zip(&self.row_val[span]))
                .map(|(&k, &v)| v * c[k as usize])
                .sum()
        };
        if self.nnz() < PARALLEL_MUL_ADD {
            for (i, &r) in self.rows.iter().enumerate() {
                out[r as usize] += alpha * row(i);
            }
            return;
        }
        out.par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(|(j, block)| {
                let base = j * BLOCK;
                let lo = self.rows.partition_point(|&r| (r as usize) < base);
                let hi = (self.rows).partition_point(|&r| (r as usize) < base + block.len());
                for i in lo..hi {
                    block[self.rows[i] as usize - base] += alpha * row(i);
                }
            });
    }
}

/// `SZ = SZ₀ − F C B` for the wide levels: `B` their `s` rows, `C` their `G⁻`, `F = D_keptᵀ W E`.
struct HubCorrection {
    rows: HubRows,
    f: TwoWay,
}

/// Wide levels' `s = EᵀW DZ` rows over the live coarse columns, one per channel, with its `G`.
struct HubRows {
    ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    gram: Vec<f64>,
}

impl HubRows {
    fn rank(&self) -> usize {
        self.gram.len()
    }

    /// `E`'s rows `[G s]` past its `coarse` columns, as lower triplets.
    fn lower(&self, coarse: usize) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        (0..self.rank()).flat_map(move |i| {
            let row = (coarse + i) as u32;
            let span = self.ptr[i]..self.ptr[i + 1];
            let s = (self.cols[span.clone()].iter().zip(&self.vals[span]))
                .map(move |(&c, &v)| (c, row, v));
            std::iter::once((row, row, self.gram[i])).chain(s)
        })
    }

    /// Row `i` of `C v`, from its `B` row `b`.
    fn c_mul(&self, i: usize, b: f64) -> f64 {
        let g = self.gram[i];
        if g > 0.0 {
            b / g
        } else {
            0.0
        }
    }
}

impl HubCorrection {
    /// `C B v`.
    fn cb_mul(&self, v: &[f64]) -> Vec<f64> {
        let rows = &self.rows;
        (0..rows.rank())
            .map(|i| {
                let b: f64 = (rows.ptr[i]..rows.ptr[i + 1])
                    .map(|k| rows.vals[k] * v[rows.cols[k] as usize])
                    .sum();
                rows.c_mul(i, b)
            })
            .collect()
    }

    /// `out += F C B c`.
    fn add(&self, c: &[f64], out: &mut [f64]) {
        self.f.mul_add(&self.cb_mul(c), 1.0, out);
    }

    /// `out −= (F C B)ᵀ y = Bᵀ C Fᵀ y`.
    fn sub_transpose(&self, y: &[f64], out: &mut [f64]) {
        let rows = &self.rows;
        let mut fy = vec![0.0; rows.rank()];
        self.f.t_mul(y, &mut fy);
        for (i, f) in fy.into_iter().enumerate() {
            let q = rows.c_mul(i, f);
            for k in rows.ptr[i]..rows.ptr[i + 1] {
                out[rows.cols[k] as usize] -= rows.vals[k] * q;
            }
        }
    }
}

/// A level's columns with a diagonal Gram: row `k`'s entries and each column's squared norm.
trait DiagonalGramBasis {
    /// Row `p`'s entry in column `c`.
    fn entry(&self, p: usize, c: usize) -> f64;
    /// Column `c`'s squared norm; a column with none fits nothing.
    fn norm2(&self, c: usize) -> f64;
}

/// Adds row `p`'s `entry · v` to each column's moment.
#[inline(always)]
fn add_moments(basis: &impl DiagonalGramBasis, p: usize, v: f64, moments: &mut [f64]) {
    for (c, m) in moments.iter_mut().enumerate() {
        *m += basis.entry(p, c) * v;
    }
}

/// `moments ← G⁻¹ moments`.
#[inline(always)]
fn solve(basis: &impl DiagonalGramBasis, moments: &mut [f64]) {
    for (c, m) in moments.iter_mut().enumerate() {
        let g = basis.norm2(c);
        *m = if g > 0.0 { *m / g } else { 0.0 };
    }
}

/// `v − Σ_c entry(p, c) · fit_c`.
#[inline(always)]
fn remove_fit(basis: &impl DiagonalGramBasis, p: usize, v: f64, fit: &[f64]) -> f64 {
    (fit.iter().enumerate()).fold(v, |v, (c, m)| sub_product(v, basis.entry(p, c), *m))
}

/// `v − a · b`, fused where the target has the instruction; without it `mul_add` calls libm.
#[inline(always)]
fn sub_product(v: f64, a: f64, b: f64) -> f64 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        (-a).mul_add(b, v)
    } else {
        v - a * b
    }
}
