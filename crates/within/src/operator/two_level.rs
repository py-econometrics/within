//! The two-level map `PᵀBP + Q`: one-level Schwarz `B` balanced by the coarse solve
//! `Q = Z E⁻¹ Zᵀ`, with `E = ZᵀSZ` and `P = I − SQ`.

mod rigid;
#[cfg(test)]
mod tests;

use std::sync::Arc;

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
use schwarz_precond::{Operator, SolveError, StackReq, Threads};
use serde::{Deserialize, Serialize};

use super::schwarz::FeSchwarz;
use crate::config::{CoarseCorrection, CoarseSpace, TwoLevelForm};
use crate::domain::{offsets, PreparedDesign};
use crate::CoarseDecline;

/// Entries from which rayon beats one thread: the gather `Mᵀx` pays later than the row-blocked
/// `M c`.
const PARALLEL_T_MUL: usize = 1 << 18;
const PARALLEL_MUL_ADD: usize = 1 << 16;

/// A coarse direction keeping less than this share of its energy is null: a column of `Z` once
/// the eliminated term is fitted, or a pivot of `E` at unit diagonal.
const NULL_ENERGY_SHARE: f64 = 1e-10;

/// `PᵀBP + Q` with `P = I − SQ`: exact on the coarse space, `M S Z = Z`.
#[derive(Clone, Debug, Deserialize)]
#[serde(try_from = "TwoLevelWire")]
pub(crate) struct TwoLevelMap {
    one_level: FeSchwarz,
    coarse: Arc<CoarseSolve>,
}

/// A sparse matrix's columns: pointers, row indices, values.
type Columns = (Vec<usize>, Vec<u32>, Vec<f64>);

/// The one-level map, `Z`, `SZ`, the hubs' rows and `F`, and `E`'s coarse block: each dimension
/// once, faer's factor having no serde.
#[derive(Deserialize)]
struct TwoLevelWire(
    FeSchwarz,
    Columns,
    Columns,
    Option<(HubRows, Columns)>,
    Vec<(u32, u32, f64)>,
);

impl Serialize for TwoLevelMap {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let c = &*self.coarse;
        let hubs = c.hubs.as_ref().map(|h| (&h.rows, h.f.columns()));
        let block = &c.factor.block;
        (&self.one_level, c.z.columns(), c.sz.columns(), hubs, block).serialize(serializer)
    }
}

impl TryFrom<TwoLevelWire> for TwoLevelMap {
    type Error = &'static str;

    fn try_from(wire: TwoLevelWire) -> Result<Self, Self::Error> {
        let TwoLevelWire(one_level, z, sz, hubs, block) = wire;
        let n = one_level.nrows();
        let z = TwoWay::from_wire(n, z)?;
        let m = z.ncols();
        let sz = TwoWay::from_wire(n, sz)?;
        let hubs = (hubs.map(|(rows, f)| {
            let f = TwoWay::from_wire(n, f)?;
            let fits = rows.ptr.len() == rows.rank() + 1
                && compressed(&rows.ptr, &rows.cols, m)
                && rows.vals.len() == rows.cols.len()
                && f.ncols() == rows.rank();
            (fits.then_some(HubCorrection { rows, f }))
                .ok_or("the hub correction does not fit the coarse space")
        }))
        .transpose()?;
        let dim = m + hubs.as_ref().map_or(0, |h| h.rows.rank());
        let lower = (block.iter()).all(|&(c, r, v)| c <= r && (r as usize) < m && v.is_finite());
        // faer indexes with i32.
        if sz.ncols() != m || !lower || dim > i32::MAX as usize {
            return Err("E's coarse block does not fit Z's columns");
        }
        let rows = hubs.as_ref().map(|h| &h.rows);
        let factor =
            CoarseFactor::new(m, block, rows, usize::MAX).map_err(|_| "E does not factor")?;
        Ok(Self {
            one_level,
            coarse: Arc::new(CoarseSolve {
                z,
                sz,
                hubs,
                factor,
            }),
        })
    }
}

/// `one_level` with the coarse correction `coarse` asks for over the unknowns, unless `prepared`
/// has no such coarse space.
pub(crate) fn build_two_level(
    prepared: &PreparedDesign<'_>,
    one_level: &FeSchwarz,
    coarse: &CoarseCorrection,
) -> Result<TwoLevelMap, CoarseDecline> {
    let CoarseCorrection {
        space: CoarseSpace::RigidComponents,
        form: TwoLevelForm::Balanced,
        build: _,
    } = coarse;
    Ok(TwoLevelMap {
        one_level: one_level.clone(),
        coarse: Arc::new(rigid::build(prepared)?),
    })
}

impl Operator for TwoLevelMap {
    fn nrows(&self) -> usize {
        self.one_level.nrows()
    }

    fn ncols(&self) -> usize {
        self.one_level.ncols()
    }

    fn scratch(&self, columns: usize, threads: Threads) -> StackReq {
        self.one_level.scratch(columns, threads)
    }

    /// `Z E⁻¹ Zᵀx + (I − Z E⁻¹ (SZ)ᵀ) B (I − SZ E⁻¹ Zᵀ) x`.
    fn apply(
        &self,
        x: &[f64],
        y: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        let coarse = &self.coarse;
        let m = coarse.z.ncols();
        let mut c = vec![0.0; m];
        coarse.z.t_mul(x, &mut c);
        coarse.factor.solve(&mut c);
        let mut t = x.to_vec();
        coarse.sub_sz(&c, &mut t);
        self.one_level.apply(&t, y, threads, stack)?;
        let mut d = vec![0.0; m];
        coarse.sz_t_mul(y, &mut d);
        coarse.factor.solve(&mut d);
        for (c, d) in c.iter_mut().zip(&d) {
            *c -= d;
        }
        coarse.z.mul_add(&c, 1.0, y);
        Ok(())
    }

    fn apply_adjoint(
        &self,
        x: &[f64],
        y: &mut [f64],
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), SolveError> {
        self.apply(x, y, threads, stack)
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

/// The LDLᵀ of `D E D`, `D` scaling `E` to a unit diagonal.
struct CoarseFactor {
    /// `E`'s merged lower triplets over the coarse columns: the wire carries them.
    block: Vec<(u32, u32, f64)>,
    symbolic: SymbolicCholesky<u32>,
    l_values: Vec<f64>,
    scale: Vec<f64>,
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
    /// `out −= SZ c`.
    fn sub_sz(&self, c: &[f64], out: &mut [f64]) {
        self.sz.mul_add(c, -1.0, out);
        if let Some(hubs) = &self.hubs {
            hubs.add(c, out);
        }
    }

    /// `out = SZᵀ y`.
    fn sz_t_mul(&self, y: &[f64], out: &mut [f64]) {
        self.sz.t_mul(y, out);
        if let Some(hubs) = &self.hubs {
            hubs.sub_transpose(y, out);
        }
    }
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
        let block = (lower.iter())
            .filter(|e| (e.1 as usize) < coarse)
            .copied()
            .collect();
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
            block,
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

/// Whether `ptr` cuts `idx` into runs from 0 to its end, every index below `bound`.
fn compressed(ptr: &[usize], idx: &[u32], bound: usize) -> bool {
    ptr.first() == Some(&0)
        && ptr.last() == Some(&idx.len())
        && ptr.is_sorted()
        && idx.iter().all(|&r| (r as usize) < bound)
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

    /// From the wire's columns over the one-level map's `n` rows.
    fn from_wire(n: usize, (col_ptr, col_idx, col_val): Columns) -> Result<Self, &'static str> {
        let valid = compressed(&col_ptr, &col_idx, n)
            && col_val.len() == col_idx.len()
            && col_ptr.len() <= u32::MAX as usize
            && n < u32::MAX as usize;
        match valid {
            true => Ok(Self::from_columns(n, col_ptr, col_idx, col_val)),
            false => Err("a sparse matrix's columns are malformed"),
        }
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

    fn columns(&self) -> (&[usize], &[u32], &[f64]) {
        (&self.col_ptr, &self.col_idx, &self.col_val)
    }

    fn nnz(&self) -> usize {
        self.col_idx.len()
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
#[derive(Serialize, Deserialize)]
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
