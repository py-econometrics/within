//! PROTOTYPE(#612): `S = Σ_g S_g` assembled in the solve basis, for algebraic coarse spaces. Its
//! block over the large kept terms' unknowns is summed from each eliminated level's
//! `S_g = Ã_gᵀÃ_g`, `Ã_g = (I − P_g) A_g` on the level's rows; its whole columns at the small
//! terms' unknowns come from applying `S`, so no level emits the dense coupling to them.

use rayon::prelude::*;
use schwarz_precond::{MemBuffer, MemStack, Operator};

use super::rigid::Rows;
use super::NULL_ENERGY_SHARE;
use super::{merged, CoarseDecline, CoarseFactor, CoarseSolve, DiagonalGramBasis, TwoWay};
use crate::system::blocks::{DynamicKept, KeptTerms};
use crate::system::{ReducedDesign, SystemMatrix};

/// A kept term with at most this many unknowns is small: its columns of `S` are stored whole.
const SMALL_TERM: usize = 512;
/// The small terms' unknowns, together, at most.
const SMALL_TOTAL: usize = 1024;

/// `S` in the solve basis: large × large as symmetric CSR, the small unknowns' columns dense.
pub(crate) struct AssembledS {
    pub(super) n: usize,
    /// Per unknown, its index among the small ones, or `u32::MAX`.
    small_of: Vec<u32>,
    /// The small unknowns, ascending.
    pub(super) small: Vec<u32>,
    /// `S[:, small[s]]` at `s · n`.
    dense: Vec<f64>,
    /// The large × large block by row, both triangles; rows of small unknowns are empty.
    pub(super) row_ptr: Vec<usize>,
    pub(super) col_idx: Vec<u32>,
    pub(super) val: Vec<f64>,
    /// `diag(A_kᵀA_k)`: each unknown's energy before the elimination.
    pub(super) pre: Vec<f64>,
    /// `diag(S)`, `0` where the elimination absorbs the column.
    pub(super) diag: Vec<f64>,
}

impl std::fmt::Debug for AssembledS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssembledS")
            .field("n", &self.n)
            .field("small", &self.small.len())
            .field("nnz", &self.col_idx.len())
            .finish()
    }
}

impl AssembledS {
    pub(crate) fn assemble(design: &ReducedDesign<'_>) -> Self {
        let split = &design.split;
        let n = split.unknowns().len();
        let diag = design.diag_s.values().to_vec();
        let small = small_unknowns(design);
        let mut small_of = vec![u32::MAX; n];
        for (s, &u) in small.iter().enumerate() {
            small_of[u as usize] = s as u32;
        }
        let lower = lower_triplets(design, |u| small_of[u] == u32::MAX && diag[u] != 0.0);
        let mut counts = vec![0usize; n + 1];
        for &(c, r, _) in &lower {
            counts[r as usize + 1] += 1;
            if c != r {
                counts[c as usize + 1] += 1;
            }
        }
        for i in 1..=n {
            counts[i] += counts[i - 1];
        }
        let row_ptr = counts.clone();
        let mut cursor = counts;
        let (mut col_idx, mut val) = (vec![0u32; row_ptr[n]], vec![0.0; row_ptr[n]]);
        for &(c, r, v) in &lower {
            let at = &mut cursor[r as usize];
            (col_idx[*at], val[*at]) = (c, v);
            *at += 1;
            if c != r {
                let at = &mut cursor[c as usize];
                (col_idx[*at], val[*at]) = (r, v);
                *at += 1;
            }
        }
        drop(lower);
        let kept = DynamicKept::new(split);
        let mut pre = vec![0.0; n];
        let sqrt_w = split.weights.sqrt();
        for i in 0..split.n_obs() {
            let w = sqrt_w.map_or(1.0, |s| s[i] * s[i]);
            kept.slots(i, |u, d| pre[u] += w * d * d);
        }
        let dense = dense_columns(design, &small);
        Self {
            n,
            small_of,
            small,
            dense,
            row_ptr,
            col_idx,
            val,
            pre,
            diag,
        }
    }

    /// Whether unknown `u` is small.
    pub(super) fn is_small(&self, u: usize) -> bool {
        self.small_of[u] != u32::MAX
    }

    /// `f(row, S[row, u])` over column `u`'s stored entries.
    pub(super) fn column(&self, u: usize, mut f: impl FnMut(usize, f64)) {
        match self.small_of[u] {
            u32::MAX => {
                for k in self.row_ptr[u]..self.row_ptr[u + 1] {
                    f(self.col_idx[k] as usize, self.val[k]);
                }
                for (s, &r) in self.small.iter().enumerate() {
                    let v = self.dense[s * self.n + u];
                    if v != 0.0 {
                        f(r as usize, v);
                    }
                }
            }
            s => {
                let col = &self.dense[s as usize * self.n..][..self.n];
                for (r, &v) in col.iter().enumerate() {
                    if v != 0.0 {
                        f(r, v);
                    }
                }
            }
        }
    }

    /// `y = S x`.
    pub(super) fn mul(&self, x: &[f64], y: &mut [f64]) {
        y.par_iter_mut().enumerate().for_each(|(i, y)| {
            let mut acc = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                acc += self.val[k] * x[self.col_idx[k] as usize];
            }
            if self.is_small(i) {
                let s = self.small_of[i] as usize;
                acc = (self.dense[s * self.n..][..self.n].iter().zip(x))
                    .map(|(a, b)| a * b)
                    .sum();
            } else {
                for (s, &u) in self.small.iter().enumerate() {
                    acc += self.dense[s * self.n + i] * x[u as usize];
                }
            }
            *y = acc;
        });
    }

    /// `S z` for a sparse `z`, merged by row in input order.
    pub(super) fn mul_sparse(&self, z: &[(u32, f64)]) -> Vec<(u32, f64)> {
        let mut out = Vec::new();
        for &(u, zu) in z {
            self.column(u as usize, |r, v| out.push((r as u32, zu * v)));
        }
        out.sort_by_key(|e| e.0);
        (out.chunk_by(|a, b| a.0 == b.0))
            .filter_map(|run| {
                let v = run.iter().fold(0.0, |v, e| v + e.1);
                (v != 0.0).then_some((run[0].0, v))
            })
            .collect()
    }
}

/// The small kept terms' unknowns, ascending: none when there are too many.
pub(super) fn small_unknowns(design: &ReducedDesign<'_>) -> Vec<u32> {
    let mut small: Vec<u32> = (design.split.unknowns().blocks())
        .filter(|(_, at)| at.len() <= SMALL_TERM)
        .flat_map(|(_, at)| at.start as u32..at.end as u32)
        .collect();
    if small.len() > SMALL_TOTAL {
        small.clear();
    }
    small
}

/// `S`'s whole columns at `small`, `s · n`, from applying `S`.
pub(super) fn dense_columns(design: &ReducedDesign<'_>, small: &[u32]) -> Vec<f64> {
    let n = design.split.unknowns().len();
    let op = SystemMatrix::new(design);
    let mut buf = MemBuffer::new(op.scratch());
    let mut dense = vec![0.0; small.len() * n];
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; design.split.n_obs()]);
    for (s, &u) in small.iter().enumerate() {
        x[u as usize] = 1.0;
        let stack = MemStack::new(&mut buf);
        op.apply_with(&x, &mut y, stack).expect("S applies");
        let stack = MemStack::new(&mut buf);
        (op.apply_adjoint_with(&y, &mut dense[s * n..(s + 1) * n], stack)).expect("S applies");
        x[u as usize] = 0.0;
    }
    dense
}

/// Each eliminated level's `S_g = Ã_gᵀÃ_g`, `Ã_g = (I − P_g) A_g` on the level's rows.
pub(super) struct LevelGrams<'p> {
    rows: Rows<'p>,
    kept: DynamicKept<'p>,
    p: usize,
    inverse: &'p [f64],
}

impl<'p> LevelGrams<'p> {
    pub(super) fn new(design: &'p ReducedDesign<'_>) -> Self {
        let split = &design.split;
        let rows = Rows::new(split);
        Self {
            p: rows.width(),
            rows,
            kept: DynamicKept::new(split),
            inverse: &split.eliminated_inverse,
        }
    }

    pub(super) fn n_levels(&self) -> usize {
        self.rows.n_levels()
    }

    /// The unknowns `keep` selects that level `g` touches, ascending, and `S_g` over them,
    /// row-major.
    pub(super) fn gram(&self, g: usize, keep: &impl Fn(usize) -> bool) -> (Vec<u32>, Vec<f64>) {
        let (rows, kept, p) = (&self.rows, &self.kept, self.p);
        let basis = rows.level(g);
        let mut cols: Vec<u32> = Vec::new();
        let mut n_rows = 0;
        for (_, r) in basis.weighted() {
            n_rows += 1;
            kept.slots(r, |u, d| {
                if d != 0.0 && keep(u) {
                    cols.push(u as u32);
                }
            });
        }
        cols.sort_unstable();
        cols.dedup();
        let t = cols.len();
        if t == 0 {
            return (cols, Vec::new());
        }
        let mut a = vec![0.0; n_rows * t];
        let mut e = vec![0.0; n_rows * p];
        for (k, (at, r)) in basis.weighted().enumerate() {
            let root = rows.root(r);
            kept.slots(r, |u, d| {
                if d != 0.0 && keep(u) {
                    let j = cols.binary_search(&(u as u32)).expect("collected");
                    a[k * t + j] += root * d;
                }
            });
            for c in 0..p {
                e[k * p + c] = basis.entry(at, c);
            }
        }
        // Ã = (I − P_g) A: the level's columns are orthogonal, so one at a time.
        let inv = &self.inverse[g * p..(g + 1) * p];
        let mut coef = vec![0.0; t];
        for c in 0..p {
            if inv[c] == 0.0 {
                continue;
            }
            coef.iter_mut().for_each(|x| *x = 0.0);
            for k in 0..n_rows {
                let ek = e[k * p + c];
                for j in 0..t {
                    coef[j] += ek * a[k * t + j];
                }
            }
            coef.iter_mut().for_each(|x| *x *= inv[c]);
            for k in 0..n_rows {
                let ek = e[k * p + c];
                for j in 0..t {
                    a[k * t + j] -= ek * coef[j];
                }
            }
        }
        let mut gram = vec![0.0; t * t];
        for i in 0..t {
            for j in 0..=i {
                let v: f64 = (0..n_rows).map(|k| a[k * t + i] * a[k * t + j]).sum();
                gram[i * t + j] = v;
                gram[j * t + i] = v;
            }
        }
        (cols, gram)
    }
}

/// `S`'s lower triangle over the unknowns `keep` selects, `(column, row, value)` sorted, summed
/// from each eliminated level's `S_g` in level order.
pub(super) fn lower_triplets(
    design: &ReducedDesign<'_>,
    keep: impl Fn(usize) -> bool + Sync,
) -> Vec<(u32, u32, f64)> {
    let levels = LevelGrams::new(design);
    let level_lower = |g: usize| -> Vec<(u32, u32, f64)> {
        let (cols, gram) = levels.gram(g, &keep);
        let t = cols.len();
        let mut lower = Vec::with_capacity(t * (t + 1) / 2);
        for i in 0..t {
            for j in 0..=i {
                let v = gram[i * t + j];
                if v != 0.0 {
                    lower.push((cols[j], cols[i], v));
                }
            }
        }
        lower
    };
    // Levels in chunks, each chunk's triplets merged into the running sum: the triplets of every
    // level at once would hold `Σ_g t_g²/2` entries.
    let n_levels = levels.n_levels();
    let chunk = (n_levels / 16).max(4096);
    let mut lower: Vec<(u32, u32, f64)> = Vec::new();
    for start in (0..n_levels).step_by(chunk) {
        let end = (start + chunk).min(n_levels);
        let part: Vec<(u32, u32, f64)> = (start..end)
            .into_par_iter()
            .flat_map_iter(level_lower)
            .collect();
        lower = merge_sum(lower, merged(part));
    }
    lower
}

/// The two-level coarse solve over the sparse columns `columns` of `Z`, each `(unknown, value)`,
/// with `SZ` and `E = ZᵀSZ` from the assembled `S`; columns with no energy once the eliminated term
/// is fitted are dropped. Declines when none is left, or past `budget` factor values.
pub(super) fn coarse_from_columns(
    s: &AssembledS,
    columns: Vec<Vec<(u32, f64)>>,
    budget: usize,
) -> Result<CoarseSolve, CoarseDecline> {
    let n = s.n;
    let images: Vec<Vec<(u32, f64)>> = columns.par_iter().map(|z| s.mul_sparse(z)).collect();
    let live: Vec<usize> = (0..columns.len())
        .filter(|&c| {
            let z = &columns[c];
            let energy = sparse_dot(z, &images[c]);
            let pre: f64 = z.iter().map(|&(u, v)| v * v * s.pre[u as usize]).sum();
            energy > NULL_ENERGY_SHARE * pre
        })
        .collect();
    if live.is_empty() {
        return Err(CoarseDecline::NoCoarseSpace);
    }
    let m = live.len();
    let z_parts = vec![(live.iter().enumerate())
        .flat_map(|(k, &c)| columns[c].iter().map(move |&(u, v)| (k as u32, u, v)))
        .collect::<Vec<_>>()];
    let z = TwoWay::from_parts(n, m, z_parts);
    let sz_parts = vec![(live.iter().enumerate())
        .flat_map(|(k, &c)| images[c].iter().map(move |&(u, v)| (k as u32, u, v)))
        .collect::<Vec<_>>()];
    drop(columns);
    let sz = TwoWay::from_parts(n, m, sz_parts);
    drop(images);
    // `E = ZᵀSZ`, column by column into a dense accumulator: pushing each product would hold
    // `Σ_c Σ_{u ∈ SZ_c} |Z_u|` entries at once.
    let lower: Vec<(u32, u32, f64)> = (0..m)
        .into_par_iter()
        .map_init(
            || (vec![0.0f64; m], Vec::<u32>::new()),
            |(acc, touched), c| {
                for k in sz.col_ptr[c]..sz.col_ptr[c + 1] {
                    let (u, v) = (sz.col_idx[k], sz.col_val[k]);
                    if let Some((lo, hi)) = z.row_span(u) {
                        for q in lo..hi {
                            let i = z.row_idx[q] as usize;
                            if i >= c {
                                if acc[i] == 0.0 {
                                    touched.push(i as u32);
                                }
                                acc[i] += z.row_val[q] * v;
                            }
                        }
                    }
                }
                touched.sort_unstable();
                touched.dedup();
                let out: Vec<(u32, u32, f64)> = (touched.iter())
                    .map(|&i| (c as u32, i, std::mem::take(&mut acc[i as usize])))
                    .filter(|e| e.2 != 0.0)
                    .collect();
                touched.clear();
                out
            },
        )
        .flatten_iter()
        .collect();
    let factor = CoarseFactor::new(m, lower, None, budget)?;
    Ok(CoarseSolve {
        z,
        sz,
        hubs: None,
        factor,
    })
}

/// `aᵀb` of two sparse vectors sorted by index.
fn sparse_dot(a: &[(u32, f64)], b: &[(u32, f64)]) -> f64 {
    let mut a: Vec<(u32, f64)> = a.to_vec();
    a.sort_by_key(|e| e.0);
    let (mut i, mut j, mut acc) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                acc += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    acc
}

/// Two `(column, row)`-sorted triplet lists summed into one; `a`'s value first where both hold an
/// entry.
fn merge_sum(a: Vec<(u32, u32, f64)>, b: Vec<(u32, u32, f64)>) -> Vec<(u32, u32, f64)> {
    if a.is_empty() {
        return b;
    }
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let ka = a.get(i).map(|e| (e.0, e.1));
        let kb = b.get(j).map(|e| (e.0, e.1));
        match (ka, kb) {
            (Some(x), Some(y)) if x == y => {
                out.push((x.0, x.1, a[i].2 + b[j].2));
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x < y => {
                out.push(a[i]);
                i += 1;
            }
            (Some(_), None) => {
                out.push(a[i]);
                i += 1;
            }
            _ => {
                out.push(b[j]);
                j += 1;
            }
        }
    }
    out.retain(|e| e.2 != 0.0);
    out.shrink_to_fit();
    out
}
