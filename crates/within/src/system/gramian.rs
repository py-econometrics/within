//! What the preconditioners read of `S = A_rᵀA_r = G_kk − G_ke G_ee⁻¹ G_ek`: `diag(S)` from
//! `A_r`'s columns, and local operators approximating its blocks from `G`'s pair factors.

use std::ops::Range;

use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1, Side};
use approx_chol::{ApproxInverse, Config, ScalingPolicy};

use super::{ReducedDesign, SplitDesign, TermBlock};
use crate::design::{Channel, ChannelPair};

/// What the preconditioners read of `S`: `diag(S)`, and local operators on its channel pairs,
/// each formed from a pair block of `G = D̃ᵀ W D̃`.
#[derive(Clone, Copy)]
pub(crate) struct Gramian<'p> {
    pub(super) design: &'p ReducedDesign<'p>,
}

/// A local operator on the channel pair `pair` of two terms, from `G`'s pair factor `V`. Two kept
/// channels `q, r` get `G_qr`: `S`'s block there without `e`'s Schur term `G_qe G_ee⁻¹ G_er`. A
/// kept channel `q` with one of `e`'s, `c`, gets `G_qc`'s complement onto `q`: `Δ_q` less `c`'s
/// share of that term.
pub(crate) struct PairBlock<'p> {
    pub(crate) pair: ChannelPair,
    /// `G`'s block on `pair`: `G = V Vᵀ`, one column per observation.
    factor: BipartiteFactorWidth2<'p, f64>,
    shape: BlockShape,
}

impl PairBlock<'_> {
    /// The local operator's approximate inverse, formed from `G`'s block as its [`BlockShape`] says.
    pub(crate) fn factorize(
        &self,
        approx: Config,
        scaling: ScalingPolicy,
    ) -> Result<ApproxInverse, approx_chol::BuildError> {
        match self.shape {
            BlockShape::Kept { .. } => approx_chol::factorize_with(self.factor, approx, scaling),
            BlockShape::Complement { side, .. } => {
                approx_chol::factorize_complement(self.factor, side, approx, scaling)
            }
        }
    }

    /// The unknown at vertex `vertex` of the factored operator.
    pub(crate) fn unknown(&self, vertex: usize) -> usize {
        self.shape.unknown(vertex)
    }
}

/// Which operator a pair block is, `G`'s block or its complement, and where its vertices sit
/// among the unknowns.
enum BlockShape {
    /// Both channels kept: `G`'s block itself, its vertices the rows' slots, then the columns'.
    Kept {
        rows: Range<usize>,
        cols: Range<usize>,
    },
    /// One channel is `e`'s: `G`'s block complemented onto the kept channel, which is on `side`
    /// of the factor; its vertices are that channel's slots.
    Complement { kept: Range<usize>, side: Side },
}

impl BlockShape {
    /// The unknown at vertex `vertex` of the formed block.
    fn unknown(&self, vertex: usize) -> usize {
        match self {
            Self::Kept { rows, .. } if vertex < rows.len() => rows.start + vertex,
            Self::Kept { rows, cols } => cols.start + vertex - rows.len(),
            Self::Complement { kept, .. } => kept.start + vertex,
        }
    }
}

impl<'p> Gramian<'p> {
    pub(crate) fn new(design: &'p ReducedDesign<'p>) -> Self {
        Self { design }
    }

    /// `S`'s order: the unknowns, which the preconditioners map.
    pub(crate) fn n(&self) -> usize {
        self.design.unknowns().len()
    }

    /// Whether `G` has a pair block, hence a local operator: two terms, each kept channel pairing
    /// with `e`'s.
    pub(crate) fn has_pair_blocks(&self) -> bool {
        self.design.layout().n_terms() >= 2
    }

    /// Whether a channel of `G`'s pair blocks is a slope, `e`'s included.
    pub(crate) fn has_slopes(&self) -> bool {
        (self.design.layout().terms().iter()).any(|term| term.n_slopes() > 0)
    }

    /// `diag(S)` over the unknowns: each kept column's norm less its projection onto `e`'s span;
    /// 0 on a column the elimination absorbs.
    pub(crate) fn diagonal(&self) -> Vec<f64> {
        self.design.diag_s.values().to_vec()
    }

    /// The local operators: every pair of kept channels of different terms, then every kept
    /// channel with each of `e`'s. Same-term channels are orthogonal, so their block is 0.
    #[allow(dead_code)] // PROTOTYPE(#614): the build reads `pair_blocks_in`
    pub(crate) fn pair_blocks(&self) -> Vec<PairBlock<'p>> {
        self.pair_blocks_in(None, false)
    }

    /// PROTOTYPE(#614): the pair blocks with `raw`'s loadings in place of the orthogonal ones,
    /// and with `same`, a block per pair of one kept term's intercept and slope.
    pub(crate) fn pair_blocks_in<'r>(
        &self,
        raw: Option<&'r RawSlopes>,
        same: bool,
    ) -> Vec<PairBlock<'r>>
    where
        'p: 'r,
    {
        let layout = self.design.layout();
        let channels: Vec<(Channel, Range<usize>)> =
            self.design.unknowns().channels(layout).collect();
        let mut blocks = Vec::new();
        for (i, (rows, at_rows)) in channels.iter().enumerate() {
            for (cols, at_cols) in &channels[i + 1..] {
                if cols.term != rows.term || same {
                    let pair = ChannelPair {
                        rows: *rows,
                        cols: *cols,
                    };
                    let (rows, cols) = (at_rows.clone(), at_cols.clone());
                    blocks.push(self.block(pair, BlockShape::Kept { rows, cols }, raw));
                }
            }
        }
        for (channel, at) in &channels {
            for e in layout.channels(self.design.split.eliminated()) {
                let (pair, side) = if e.term < channel.term {
                    let (rows, cols) = (e, *channel);
                    (ChannelPair { rows, cols }, Side::Right)
                } else {
                    let (rows, cols) = (*channel, e);
                    (ChannelPair { rows, cols }, Side::Left)
                };
                let kept = at.clone();
                blocks.push(self.block(pair, BlockShape::Complement { kept, side }, raw));
            }
        }
        blocks
    }

    /// PROTOTYPE(#614): each term's one slope in the caller's basis, `z − c` with `c` from
    /// `PROBE_ORIGIN` (`zero`, `mean`, or a number), 0 on a level that drops it, read back
    /// through the level's `u = f·z − s`; and per kept term with an intercept, the map
    /// `M = [[1, f·c − s], [0, f]]` from solve to caller coefficients. Terms with more slopes, or
    /// a slope and no intercept, keep the orthogonal basis.
    pub(crate) fn raw_slopes(&self) -> RawSlopes {
        let split = &self.design.split;
        let layout = self.design.layout();
        let channels: Vec<(Channel, Range<usize>)> =
            self.design.unknowns().channels(layout).collect();
        let origin = std::env::var("PROBE_ORIGIN").unwrap_or_default();
        let mut loadings = Vec::new();
        let mut maps = Vec::new();
        for t in 0..layout.n_terms() {
            let term = split.term(t);
            let tl = term.layout();
            if !(tl.intercept() && tl.n_slopes() == 1) {
                loadings.push(None);
                continue;
            }
            let slope = tl.slope_column(0);
            let (u, levels) = (term.loading(slope).expect("a slope column"), term.levels());
            let fs: Vec<(f64, f64)> = (0..tl.n_levels())
                .map(|l| {
                    let (f, s, _) = split.slopes[t].level_map(l);
                    (f[0], s[0])
                })
                .collect();
            let z = |i: usize| {
                let (f, s) = fs[levels[i] as usize];
                if f != 0.0 {
                    (u[i] + s) / f
                } else {
                    f64::NAN
                }
            };
            let c = match origin.as_str() {
                "" | "zero" => 0.0,
                "mean" => {
                    let (sum, count) = (0..u.len())
                        .map(z)
                        .filter(|v| v.is_finite())
                        .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
                    sum / count.max(1) as f64
                }
                other => other.parse().expect("PROBE_ORIGIN: zero, mean or a number"),
            };
            let raw: Vec<f64> = (0..u.len())
                .map(|i| match z(i) {
                    v if v.is_finite() => v - c,
                    _ => 0.0,
                })
                .collect();
            loadings.push(Some(raw));
            if t != split.eliminated() {
                let at = |column: usize| {
                    let find = channels
                        .iter()
                        .find(|(ch, _)| ch.term == t && ch.column == column);
                    find.expect("a kept channel").1.start
                };
                let icept = at(tl.intercept_column().expect("an intercept"));
                let gf = fs.iter().map(|&(f, s)| (f, f * c - s)).collect();
                maps.push(RawMap {
                    intercept: icept,
                    slope: at(slope),
                    levels: gf,
                });
            }
        }
        RawSlopes { loadings, maps }
    }

    /// `pair`'s block of `G` as its factor `V`, `√w_k (z_q(k) e_{a_k} + z_r(k) e_{n_q + b_k})`.
    fn block<'r>(
        &self,
        pair: ChannelPair,
        shape: BlockShape,
        raw: Option<&'r RawSlopes>,
    ) -> PairBlock<'r>
    where
        'p: 'r,
    {
        let (rows, cols) = (
            self.design.split.term(pair.rows.term),
            self.design.split.term(pair.cols.term),
        );
        let side = |term: &TermBlock<'p>, t: usize, column: usize| -> FactorWidth1<'r, f64> {
            let (n, levels) = (term.layout().n_levels(), term.levels());
            let raw = raw.and_then(|r| r.loadings[t].as_deref());
            match (term.loading(column), raw) {
                (Some(_), Some(z)) => FactorWidth1::valued(n, levels, z),
                (Some(z), None) => FactorWidth1::valued(n, levels, z),
                (None, _) => FactorWidth1::new(n, levels),
            }
            .expect("levels index their term")
        };
        let (left, right) = (
            side(&rows, pair.rows.term, pair.rows.column),
            side(&cols, pair.cols.term, pair.cols.column),
        );
        let factor = match self.design.split.weights.sqrt() {
            Some(s) => BipartiteFactorWidth2::scaled(left, right, s),
            None => BipartiteFactorWidth2::new(left, right),
        }
        .expect("both sides read the same observations");
        PairBlock {
            pair,
            factor,
            shape,
        }
    }
}

/// PROTOTYPE(#614): the raw slope basis Schwarz is built in.
pub(crate) struct RawSlopes {
    /// Per term, its one slope as `z − c`; `None` keeps the orthogonal loading.
    loadings: Vec<Option<Vec<f64>>>,
    /// The kept terms' solve-to-caller maps.
    pub(crate) maps: Vec<RawMap>,
}

/// PROTOTYPE(#614): one kept term's `M = [[1, g], [0, f]]` per level, from its solve coefficients
/// (intercept, slope) to the caller's; identity where `f = 0` (a dropped slope).
#[derive(Clone, Debug)]
pub(crate) struct RawMap {
    pub(crate) intercept: usize,
    pub(crate) slope: usize,
    /// Per level `(f, g)`.
    pub(crate) levels: Vec<(f64, f64)>,
}

impl RawMap {
    /// `r ← M⁻ᵀ r`.
    pub(crate) fn inverse_adjoint(&self, r: &mut [f64]) {
        for (l, &(f, g)) in self.levels.iter().enumerate() {
            if f != 0.0 {
                let (a, b) = (self.intercept + l, self.slope + l);
                r[b] = (r[b] - g * r[a]) / f;
            }
        }
    }

    /// `z ← M⁻¹ z`.
    pub(crate) fn inverse(&self, z: &mut [f64]) {
        for (l, &(f, g)) in self.levels.iter().enumerate() {
            if f != 0.0 {
                let (a, b) = (self.intercept + l, self.slope + l);
                z[b] /= f;
                z[a] -= g * z[b];
            }
        }
    }
}

/// `diag(S) = diag(A_rᵀA_r)` per unknown, 0 on a column the elimination absorbs, and a floor under
/// `‖A_r‖₂` that survives the cancellation in it.
pub(super) struct SchurDiagonal {
    values: Vec<f64>,
    floor: f64,
}

impl SchurDiagonal {
    /// `diag(S)_j = Σ_l (‖a_{j,l}‖² − Σ_c (a_{c,l}ᵀ a_j)² / g_{c,l})` over `e`'s levels `l`: each
    /// level's part of `a_j` less its energy in that level's span. A level's part is 0 within
    /// `rows·ε` of its own norm, so a light level that identifies `a_j` is not lost in a heavy
    /// one's cancellation.
    pub(super) fn of(design: &SplitDesign<'_>) -> Self {
        let n = design.unknowns().len();
        let mut energies = vec![0.0; 2 * n];
        // Measured, absorbed slopes round to at most 0.04·rows·ε here, to 10⁶ rows, weighted or
        // not; the near-combination slope the solver must keep lies 11ε above its 8-row span. A
        // worst-case bound, `(2√w + 3 + 6v)·rows·ε/2` with CholeskyQR2's orthogonality, would
        // mask it.
        let slack = |rows: usize, own: f64| rows as f64 * f64::EPSILON * own;
        design.dynamic().column_energies(&mut energies, &slack);
        let floor = energies[n..].iter().fold(0.0f64, |m, &d| m.max(d)).sqrt();
        energies.truncate(n);
        energies.shrink_to_fit();
        Self {
            values: energies,
            floor,
        }
    }

    pub(super) fn values(&self) -> &[f64] {
        &self.values
    }

    /// `√max_j` of column `j`'s energy less its slack: `≤ ‖A_r e_j‖ ≤ ‖A_r‖₂` while the rounding
    /// stays inside the slack, where a near-absorbed column's own `diag(S)` may not.
    pub(super) fn floor(&self) -> f64 {
        self.floor
    }
}
