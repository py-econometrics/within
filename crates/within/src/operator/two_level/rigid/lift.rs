//! The coarse columns' image `U = ÃZ`, fitted one eliminated level at a time, and the coarse
//! solve it determines: `Z`, `E = UᵀU` and its factor, `SZ`, and the hub coupling `F`.

use rayon::prelude::*;

use super::super::{
    merged, CoarseFactor, CoarseSolve, HubCorrection, HubRows, TwoWay, NULL_ENERGY_SHARE,
};
use super::glue::Gauges;
use super::Rows;
use crate::domain::{cut, offsets, PreparedDesign, TermRows, Unknowns};
use crate::linalg::{add_moments, remove_fit, solve, DiagonalGramBasis};
use crate::CoarseDecline;

/// The coarse solve over the columns of `gauges`; declines when no column keeps energy once the
/// eliminated term is fitted, or when the factor of `E` exceeds `n_obs` values.
pub(super) fn coarse_solve(
    prepared: &PreparedDesign<'_>,
    unknowns: Unknowns,
    rows: &Rows<'_>,
    gauges: &Gauges<'_>,
) -> Result<CoarseSolve, CoarseDecline> {
    let design = &prepared.design;
    let kept = (0..design.n_factors())
        .filter(|&t| t != design.dominant)
        .map(|t| {
            let start = design.block_start(unknowns, t).expect("a kept term");
            (TermRows::new(&prepared.term(t)), start)
        })
        .collect();
    let lift = Lift { rows, gauges, kept };
    let mut fits: Vec<LevelFit> = (0..rows.n_levels())
        .into_par_iter()
        .map(|lw| lift.fit(lw))
        .collect();
    let live = Live::new(&fits, gauges, rows.width())?;
    let n = design.n_unknowns(unknowns);
    let eliminated = design.block_start(unknowns, design.dominant);
    let z = lift.coarse_basis(&fits, &live, eliminated, n);
    let (lower, hubs) = lift.coarse_system(&mut fits, &live);
    let factor = CoarseFactor::new(live.count + hubs.rank(), live.count, lower, design.n_obs)?;
    let sz = lift.projected_columns(&fits, &live, n);
    let f = lift.hub_coupling(&fits, n, hubs.rank());
    Ok(CoarseSolve {
        z,
        sz,
        hubs: (hubs.rank() > 0).then_some(HubCorrection { rows: hubs, f }),
        factor,
    })
}

/// The coarse columns of a level touching components `comps`: every channel of each.
fn columns(comps: &[u32], p: usize) -> impl Iterator<Item = u32> + '_ {
    (comps.iter()).flat_map(move |&c| (0..p as u32).map(move |j| c * p as u32 + j))
}

/// One eliminated level's share of `U = ÃZ` over the coarse columns its rows touch, in
/// [`columns`] order.
struct LevelFit {
    /// Per column, its `β = G⁻ Eᵀ W½ (W½ D Z)` over the `width` eliminated columns.
    beta: Vec<f64>,
    /// Taken once `E` is assembled.
    gram: LevelGram,
    coupling: Coupling,
}

/// A level's share of the column energies and of `E = UᵀU`.
#[derive(Default)]
struct LevelGram {
    /// `‖W½DZ‖²` per column.
    pre: Vec<f64>,
    /// `‖U‖²` per column.
    energy: Vec<f64>,
    /// The lower triangle of `UᵀU`; a wide level's own blocks only.
    lower: Vec<(u32, u32, f64)>,
}

/// How a level's projection enters the coarse system.
enum Coupling {
    /// Its block of `UᵀU` in `E`, its projected rows in `SZ`.
    Block,
    /// A wide level: `E` gains its rows `[G s]`, and `SZ` takes each component's own rows
    /// unprojected, the hub correction projecting them.
    Rows {
        /// `s = EᵀW DZ`, `width` per column.
        s: Vec<f64>,
        /// Its weighted rows, by component.
        by_comp: Vec<(u32, usize)>,
    },
}

impl LevelFit {
    /// A wide level's weighted rows on component `comp`; `None` for a block.
    fn wide_rows(&self, comp: usize) -> Option<impl Iterator<Item = usize> + '_> {
        let Coupling::Rows { by_comp, .. } = &self.coupling else {
            return None;
        };
        let lo = by_comp.partition_point(|e| (e.0 as usize) < comp);
        let hi = by_comp.partition_point(|e| (e.0 as usize) <= comp);
        Some(by_comp[lo..hi].iter().map(|&(_, r)| r))
    }
}

/// The coarse columns with energy left once the eliminated term is fitted, renumbered: exact
/// nulls of `S`, where the eliminated term absorbs the gauge, need no lift.
struct Live {
    of: Vec<u32>,
    count: usize,
}

impl Live {
    fn new(fits: &[LevelFit], gauges: &Gauges<'_>, p: usize) -> Result<Self, CoarseDecline> {
        let n_columns = gauges.n_components() * p;
        let (mut pre, mut energy) = (vec![0.0f64; n_columns], vec![0.0f64; n_columns]);
        for (fit, comps) in fits.iter().zip(gauges.touched()) {
            for (c, (a, b)) in columns(comps, p).zip(fit.gram.pre.iter().zip(&fit.gram.energy)) {
                pre[c as usize] += a;
                energy[c as usize] += b;
            }
        }
        let mut count = 0u32;
        let of = (energy.iter().zip(&pre))
            .map(|(&e, &pre)| match e > NULL_ENERGY_SHARE * pre {
                true => {
                    count += 1;
                    count - 1
                }
                false => u32::MAX,
            })
            .collect();
        match count {
            0 => Err(CoarseDecline::NoCoarseSpace),
            count => Ok(Self {
                of,
                count: count as usize,
            }),
        }
    }

    /// Coarse column `c`'s live index, if it has one.
    fn get(&self, c: u32) -> Option<u32> {
        Some(self.of[c as usize]).filter(|&k| k != u32::MAX)
    }

    /// The live coarse columns with their live index, ascending.
    fn columns(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        (self.of.iter().enumerate()).filter_map(|(c, &k)| (k != u32::MAX).then_some((c, k)))
    }
}

/// `U = ÃZ` one entry at a time, and the kept terms projecting it.
struct Lift<'a> {
    rows: &'a Rows<'a>,
    gauges: &'a Gauges<'a>,
    /// Per kept term, its columns and its block start among the unknowns.
    kept: Vec<(TermRows<'a>, usize)>,
}

impl Lift<'_> {
    /// `f(unknown, loading)` over row `i`'s kept coefficients.
    fn for_each_kept(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        for (term, start) in &self.kept {
            term.for_each_column(*start, i, &mut f);
        }
    }

    /// Coarse column `c`'s `β` in level `lw`'s fit.
    fn beta<'f>(&self, fit: &'f LevelFit, lw: usize, c: usize) -> &'f [f64] {
        let p = self.rows.width();
        let k = self.gauges.touched()[lw]
            .binary_search(&((c / p) as u32))
            .expect("touched column");
        &fit.beta[(k * p + c % p) * p..][..p]
    }

    /// Level `lw`'s fit, `‖W½DZ‖²` and the lower triangle of `UᵀU` over its columns.
    ///
    /// `UᵀU = (DZ)ᵀW(DZ) − sᵀG⁻s` with `s = EᵀW DZ`: a row touches only its own component's
    /// columns, so between components only `−sᵀG⁻s` remains; within one, see
    /// [`Self::component_blocks`].
    fn fit(&self, lw: usize) -> LevelFit {
        let (gauges, p) = (self.gauges, self.rows.width());
        let comps = &gauges.touched()[lw];
        let basis = self.rows.level(lw);
        let t = comps.len() * p;
        let wide = gauges.is_wide(lw);
        let (mut pre, mut s) = (vec![0.0; t], vec![0.0; t * p]);
        let mut blocks = vec![0.0; if wide { comps.len() * p * p } else { 0 }];
        let mut lifted = vec![0.0; p];
        let mut by_comp = Vec::new();
        for (at, r) in basis.weighted() {
            let c = gauges.component(r);
            let k = comps.binary_search(&c).expect("touched component");
            for (j, v) in lifted.iter_mut().enumerate() {
                *v = gauges.lifted(r, c as usize * p + j);
                pre[k * p + j] += *v * *v;
            }
            if wide {
                by_comp.push((c, r));
                let block = &mut blocks[k * p * p..][..p * p];
                for (i, &vi) in lifted.iter().enumerate() {
                    for (j, &vj) in lifted.iter().enumerate() {
                        block[i * p + j] += vi * vj;
                    }
                }
            }
            for (j, &v) in lifted.iter().enumerate() {
                add_moments(&basis, at, v, &mut s[(k * p + j) * p..][..p]);
            }
        }
        let mut beta = s.clone();
        beta.chunks_exact_mut(p).for_each(|b| solve(&basis, b));
        let fitted =
            |a: usize, b: usize| -> f64 { (0..p).map(|k| s[a * p + k] * beta[b * p + k]).sum() };
        let blocks_u = self.component_blocks(lw, comps, &beta);
        let entry = |a: usize, b: usize| (a / p) * p * p + (a % p) * p + b % p;
        let energy = (0..t).map(|a| blocks_u[entry(a, a)]).collect();
        let column = |a: usize| comps[a / p] * p as u32 + (a % p) as u32;
        let own = if wide { &blocks } else { &blocks_u };
        let mut lower = Vec::new();
        for a in 0..t {
            let end = (a / p + 1) * p;
            (a..end).for_each(|b| lower.push((column(a), column(b), own[entry(a, b)])));
            if !wide {
                (end..t).for_each(|b| lower.push((column(a), column(b), -fitted(a, b))));
            }
        }
        let coupling = match wide {
            true => {
                by_comp.sort_unstable();
                Coupling::Rows { s, by_comp }
            }
            false => Coupling::Block,
        };
        LevelFit {
            beta,
            gram: LevelGram { pre, energy, lower },
            coupling,
        }
    }

    /// Each touched component's own `p × p` block of `UᵀU`: its rows' residual Gram plus
    /// `β_κᵀ G_¬κ β_κ` from the level's other rows. `own − fitted` would lose the digits the fit
    /// absorbs, the null modes' among them (block-angular least squares: Golub & Plemmons 1980).
    fn component_blocks(&self, lw: usize, comps: &[u32], beta: &[f64]) -> Vec<f64> {
        let (gauges, p) = (self.gauges, self.rows.width());
        let basis = self.rows.level(lw);
        let block = p * p;
        let (mut residual, mut gram) = (
            vec![0.0; comps.len() * block],
            vec![0.0; comps.len() * block],
        );
        let (mut b, mut u) = (vec![0.0; p], vec![0.0; p]);
        for (at, r) in basis.weighted() {
            let c = gauges.component(r);
            let k = comps.binary_search(&c).expect("touched component");
            b.iter_mut()
                .enumerate()
                .for_each(|(q, v)| *v = basis.entry(at, q));
            for (j, v) in u.iter_mut().enumerate() {
                let fit = &beta[(k * p + j) * p..][..p];
                *v = remove_fit(&basis, at, gauges.lifted(r, c as usize * p + j), fit);
            }
            for (i, j) in (0..p).flat_map(|i| (0..p).map(move |j| (i, j))) {
                residual[k * block + i * p + j] += u[i] * u[j];
                gram[k * block + i * p + j] += b[i] * b[j];
            }
        }
        // `G_¬κ` as prefix plus suffix sums: `G − G_κ` would cancel where `κ` holds the level.
        let mut before = vec![0.0; (comps.len() + 1) * block];
        let mut after = vec![0.0; (comps.len() + 1) * block];
        for k in 0..comps.len() {
            for e in 0..block {
                before[(k + 1) * block + e] = before[k * block + e] + gram[k * block + e];
            }
        }
        for k in (0..comps.len()).rev() {
            for e in 0..block {
                after[k * block + e] = after[(k + 1) * block + e] + gram[k * block + e];
            }
        }
        for k in 0..comps.len() {
            let rest = |q: usize, o: usize| {
                before[k * block + q * p + o] + after[(k + 1) * block + q * p + o]
            };
            for (i, j) in (0..p).flat_map(|i| (0..p).map(move |j| (i, j))) {
                let (bi, bj) = (&beta[(k * p + i) * p..][..p], &beta[(k * p + j) * p..][..p]);
                let share: f64 = (0..p)
                    .flat_map(|q| (0..p).map(move |o| (q, o)))
                    .map(|(q, o)| bi[q] * rest(q, o) * bj[o])
                    .sum();
                residual[k * block + i * p + j] += share;
            }
        }
        residual
    }

    /// `Z` over the live columns: the gauges on the anchor's unknowns and, where the system
    /// carries the eliminated term from `eliminated` on, `−β` on its unknowns, so that `A Z = U`.
    fn coarse_basis(
        &self,
        fits: &[LevelFit],
        live: &Live,
        eliminated: Option<usize>,
        n: usize,
    ) -> TwoWay {
        let (rows, p) = (self.rows, self.rows.width());
        let mut entries: Vec<(u32, u32, f64)> = (self.gauges.entries())
            .filter_map(|(c, d, v)| Some((live.get(c)?, d, v)))
            .collect();
        if let Some(start) = eliminated {
            for (lw, (fit, comps)) in fits.iter().zip(self.gauges.touched()).enumerate() {
                for (c, beta) in columns(comps, p).zip(fit.beta.chunks_exact(p)) {
                    let Some(k) = live.get(c) else {
                        continue;
                    };
                    for (a, &v) in beta.iter().enumerate().filter(|&(_, &v)| v != 0.0) {
                        entries.push((k, (start + a * rows.n_levels() + lw) as u32, -v));
                    }
                }
            }
        }
        entries.par_sort_unstable_by_key(|t| (t.0, t.1));
        TwoWay::from_parts(n, live.count, vec![entries])
    }

    /// `E`'s lower triplets over the live columns, wide levels' rows `[G s]` after them, and those
    /// rows; takes each level's share of `E` and its `s`.
    fn coarse_system(&self, fits: &mut [LevelFit], live: &Live) -> (Vec<(u32, u32, f64)>, HubRows) {
        let p = self.rows.width();
        let mut lower = Vec::new();
        let mut hubs = HubRows {
            ptr: vec![0],
            cols: Vec::new(),
            vals: Vec::new(),
            gram: Vec::new(),
        };
        for (lw, (fit, comps)) in fits.iter_mut().zip(self.gauges.touched()).enumerate() {
            lower.extend(
                (std::mem::take(&mut fit.gram).lower.into_iter())
                    .filter_map(|(a, b, v)| Some((live.get(a)?, live.get(b)?, v))),
            );
            let Coupling::Rows { s, .. } = &mut fit.coupling else {
                continue;
            };
            let s = std::mem::take(s);
            let basis = self.rows.level(lw);
            for k in 0..p {
                for (i, c) in columns(comps, p).enumerate() {
                    if let Some(c) = live.get(c) {
                        hubs.cols.push(c);
                        hubs.vals.push(s[i * p + k]);
                    }
                }
                hubs.ptr.push(hubs.cols.len());
                hubs.gram.push(basis.norm2(k));
            }
        }
        for (i, &g) in hubs.gram.iter().enumerate() {
            let row = (live.count + i) as u32;
            lower.push((row, row, g));
            let span = hubs.ptr[i]..hubs.ptr[i + 1];
            lower.extend(
                (hubs.cols[span.clone()].iter().zip(&hubs.vals[span])).map(|(&c, &v)| (c, row, v)),
            );
        }
        (lower, hubs)
    }

    /// `SZ₀ = D_keptᵀ W½ U`, one live column at a time over the levels touching its component; a
    /// wide level contributes only its own component's rows, its projection being the hub
    /// correction.
    fn projected_columns(&self, fits: &[LevelFit], live: &Live, n: usize) -> TwoWay {
        let (rows, p) = (self.rows, self.rows.width());
        let (touched, n_comps) = (self.gauges.touched(), self.gauges.n_components());
        let comp_ptr = offsets(n_comps, touched.iter().flatten().map(|&c| c as usize));
        let mut comp_levels = vec![0u32; comp_ptr[n_comps]];
        let mut cursor = comp_ptr.clone();
        for (lw, comps) in touched.iter().enumerate() {
            for &c in comps {
                let k = &mut cursor[c as usize];
                comp_levels[*k] = lw as u32;
                *k += 1;
            }
        }
        let levels_of = |comp: usize| &comp_levels[comp_ptr[comp]..comp_ptr[comp + 1]];
        let live_columns: Vec<(usize, u32)> = live.columns().collect();
        let work = |c: usize| -> usize {
            (levels_of(c / p).iter())
                .map(|&lw| match fits[lw as usize].wide_rows(c / p) {
                    Some(own) => own.count(),
                    None => rows.level(lw as usize).len(),
                })
                .sum()
        };
        let total: usize = live_columns.iter().map(|&(c, _)| work(c)).sum();
        let per = total
            .div_ceil(4 * rayon::current_num_threads().max(1))
            .max(1);
        let windows = cut(live_columns.len(), per, |i| work(live_columns[i].0));
        let parts: Vec<Vec<(u32, u32, f64)>> = windows
            .par_iter()
            .map_init(
                || (vec![0.0f64; n], vec![u32::MAX; n], Vec::<u32>::new()),
                |(acc, seen, touched_unknowns), window| {
                    let mut out = Vec::new();
                    for &(c, col) in &live_columns[window.clone()] {
                        let mut visit = |r: usize, u: f64| {
                            if u == 0.0 {
                                return;
                            }
                            let val = rows.root(r) * u;
                            self.for_each_kept(r, |d, z| {
                                if seen[d] != col {
                                    seen[d] = col;
                                    touched_unknowns.push(d as u32);
                                }
                                acc[d] += z * val;
                            });
                        };
                        for &lw in levels_of(c / p) {
                            let (lw, fit) = (lw as usize, &fits[lw as usize]);
                            if let Some(own) = fit.wide_rows(c / p) {
                                own.for_each(|r| visit(r, self.gauges.lifted(r, c)));
                                continue;
                            }
                            let (basis, beta) = (rows.level(lw), self.beta(fit, lw, c));
                            for (at, r) in basis.weighted() {
                                visit(r, remove_fit(&basis, at, self.gauges.lifted(r, c), beta));
                            }
                        }
                        touched_unknowns.sort_unstable();
                        for &d in touched_unknowns.iter() {
                            let v = std::mem::take(&mut acc[d as usize]);
                            if v != 0.0 {
                                out.push((col, d, v));
                            }
                        }
                        touched_unknowns.clear();
                    }
                    out
                },
            )
            .collect();
        TwoWay::from_parts(n, live.count, parts)
    }

    /// `F = D_keptᵀ W E` over the wide levels' eliminated columns, one per hub row.
    fn hub_coupling(&self, fits: &[LevelFit], n: usize, rank: usize) -> TwoWay {
        let (rows, p) = (self.rows, self.rows.width());
        let wide: Vec<usize> = (0..fits.len())
            .filter(|&lw| matches!(fits[lw].coupling, Coupling::Rows { .. }))
            .collect();
        let parts: Vec<Vec<(u32, u32, f64)>> = (wide.par_iter().enumerate())
            .map(|(h, &lw)| {
                let basis = rows.level(lw);
                let mut entries = Vec::new();
                for (at, r) in basis.weighted() {
                    let root = rows.root(r);
                    self.for_each_kept(r, |d, z| {
                        for a in 0..p {
                            entries.push((
                                (h * p + a) as u32,
                                d as u32,
                                root * z * basis.entry(at, a),
                            ));
                        }
                    });
                }
                merged(entries)
            })
            .collect();
        TwoWay::from_parts(n, rank, parts)
    }
}
