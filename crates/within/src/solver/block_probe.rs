//! Probe: short-recurrence block LSMR against the same code run per column, on the final rung's map.

use rayon::prelude::*;
use schwarz_precond::Operator;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.par_chunks(1 << 14)
        .zip(b.par_chunks(1 << 14))
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>())
        .sum()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    y.par_chunks_mut(1 << 14)
        .zip(x.par_chunks(1 << 14))
        .for_each(|(y, x)| y.iter_mut().zip(x).for_each(|(p, q)| *p += a * q));
}

/// Column-major `rows × s` block.
#[derive(Clone)]
struct Tall {
    rows: usize,
    s: usize,
    data: Vec<f64>,
}

/// Row-major `r × c` small matrix.
#[derive(Clone)]
struct Small {
    r: usize,
    c: usize,
    a: Vec<f64>,
}

impl Small {
    fn zeros(r: usize, c: usize) -> Self {
        Self { r, c, a: vec![0.0; r * c] }
    }
    fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        (0..n).for_each(|i| m.a[i * n + i] = 1.0);
        m
    }
    fn at(&self, i: usize, j: usize) -> f64 {
        self.a[i * self.c + j]
    }
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.a[i * self.c + j] = v;
    }
    fn t(&self) -> Self {
        let mut m = Self::zeros(self.c, self.r);
        for i in 0..self.r {
            for j in 0..self.c {
                m.set(j, i, self.at(i, j));
            }
        }
        m
    }
    fn mul(&self, o: &Small) -> Self {
        let mut m = Self::zeros(self.r, o.c);
        for i in 0..self.r {
            for k in 0..self.c {
                let x = self.at(i, k);
                for j in 0..o.c {
                    m.a[i * o.c + j] += x * o.at(k, j);
                }
            }
        }
        m
    }
    fn stack(top: &Small, bottom: &Small) -> Self {
        let mut a = top.a.clone();
        a.extend_from_slice(&bottom.a);
        Self { r: top.r + bottom.r, c: top.c, a }
    }
    fn rows(&self, from: usize, n: usize) -> Self {
        Self { r: n, c: self.c, a: self.a[from * self.c..(from + n) * self.c].to_vec() }
    }
    fn col_norm(&self, j: usize) -> f64 {
        (0..self.r).map(|i| self.at(i, j).powi(2)).sum::<f64>().sqrt()
    }
    /// Lower `L` with `self = L Lᵀ`.
    fn cholesky(&self) -> Option<Self> {
        let n = self.r;
        let mut l = Self::zeros(n, n);
        for j in 0..n {
            let d = self.at(j, j) - (0..j).map(|k| l.at(j, k).powi(2)).sum::<f64>();
            if !(d > 0.0) {
                return None;
            }
            let d = d.sqrt();
            l.set(j, j, d);
            for i in j + 1..n {
                let v = (self.at(i, j) - (0..j).map(|k| l.at(i, k) * l.at(j, k)).sum::<f64>()) / d;
                l.set(i, j, v);
            }
        }
        Some(l)
    }
    /// Inverse of a square matrix by Gauss–Jordan with partial pivoting.
    fn inv(&self) -> Self {
        let n = self.r;
        let mut a = self.clone();
        let mut b = Self::eye(n);
        for c in 0..n {
            let p = (c..n).max_by(|&i, &j| a.at(i, c).abs().total_cmp(&a.at(j, c).abs())).unwrap();
            for j in 0..n {
                a.a.swap(c * n + j, p * n + j);
                b.a.swap(c * n + j, p * n + j);
            }
            let d = a.at(c, c);
            for j in 0..n {
                a.a[c * n + j] /= d;
                b.a[c * n + j] /= d;
            }
            for i in 0..n {
                if i != c {
                    let f = a.at(i, c);
                    for j in 0..n {
                        a.a[i * n + j] -= f * a.at(c, j);
                        b.a[i * n + j] -= f * b.at(c, j);
                    }
                }
            }
        }
        b
    }
    /// Orthogonal `Q` (`r × r`) with `Q self = [R; 0]`, by Householder.
    fn qr_left(&self) -> Self {
        let (p, q) = (self.r, self.c);
        let mut h = self.clone();
        let mut qm = Self::eye(p);
        for c in 0..q {
            let norm = (c..p).map(|r| h.at(r, c).powi(2)).sum::<f64>().sqrt();
            if norm == 0.0 {
                continue;
            }
            let alpha = if h.at(c, c) > 0.0 { -norm } else { norm };
            let mut v: Vec<f64> = (c..p).map(|r| h.at(r, c)).collect();
            v[0] -= alpha;
            let vn: f64 = v.iter().map(|x| x * x).sum();
            if vn == 0.0 {
                continue;
            }
            for m in [&mut h, &mut qm] {
                for j in 0..m.c {
                    let s = 2.0 * (c..p).zip(&v).map(|(r, vi)| m.at(r, j) * vi).sum::<f64>() / vn;
                    (c..p).zip(&v).for_each(|(r, vi)| m.a[r * m.c + j] -= s * vi);
                }
            }
        }
        qm
    }
}

impl Tall {
    fn zeros(rows: usize, s: usize) -> Self {
        Self { rows, s, data: vec![0.0; rows * s] }
    }
    fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.rows..][..self.rows]
    }
    /// `self · c` for small `c` (`s × c.c`).
    fn mul(&self, c: &Small) -> Tall {
        let mut out = Tall::zeros(self.rows, c.c);
        out.data.par_chunks_mut(self.rows).enumerate().for_each(|(j, o)| {
            for i in 0..self.s {
                let f = c.at(i, j);
                if f != 0.0 {
                    o.iter_mut().zip(self.col(i)).for_each(|(o, x)| *o += f * x);
                }
            }
        });
        out
    }
    /// `self += f · o`.
    fn add(&mut self, f: f64, o: &Tall) {
        self.data.par_chunks_mut(1 << 14).zip(o.data.par_chunks(1 << 14)).for_each(|(a, b)| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a += f * b);
        });
    }
    /// `selfᵀ o`.
    fn gram(&self, o: &Tall) -> Small {
        let mut g = Small::zeros(self.s, o.s);
        for i in 0..self.s {
            for j in 0..o.s {
                g.set(i, j, dot(self.col(i), o.col(j)));
            }
        }
        g
    }
}

/// CholQR twice: `w = v Lᵀ`-style factor with `vᵀ p = I`; returns `(v, p, L)` with `w = v Lᵀ`, `p_in = p Lᵀ`.
fn cholqr2(mut w: Tall, mut p: Option<Tall>) -> Option<(Tall, Option<Tall>, Small)> {
    let mut total = Small::eye(w.s);
    for _ in 0..2 {
        let g = match &p {
            Some(p) => w.gram(p),
            None => w.gram(&w),
        };
        let g = Small { a: (0..g.r * g.c).map(|i| 0.5 * (g.a[i] + g.a[(i % g.c) * g.c + i / g.c])).collect(), ..g };
        let l = g.cholesky()?;
        let lt_inv = l.t().inv();
        w = w.mul(&lt_inv);
        p = p.map(|p| p.mul(&lt_inv));
        total = l.t().mul(&total);
    }
    Some((w, p, total.t()))
}



/// Rank-revealing block orthonormalization: `w = q Lᵀ` with `L` lower (possibly singular) and `q` full.
/// Under a metric (`p` given, `q` = M⁻¹-side), a vanished direction is replaced by `M⁻¹ r` for random `r`.
fn orthonormalize(
    mut w: Tall,
    mut p: Option<Tall>,
    m: Option<&dyn Operator>,
    seed: &mut u64,
) -> (Tall, Option<Tall>, Small) {
    let s = w.s;
    let mut r = Small::zeros(s, s);
    let norms: Vec<f64> = (0..s)
        .map(|j| match &p { Some(p) => dot(w.col(j), p.col(j)).max(0.0).sqrt(), None => dot(w.col(j), w.col(j)).sqrt() })
        .collect();
    let scale = norms.iter().fold(0.0f64, |a, &b| a.max(b));
    for j in 0..s {
        for pass in 0..2 {
            for i in 0..j {
                let c = match &p { Some(p) => dot(w.col(i), p.col(j)), None => dot(w.col(i), w.col(j)) };
                let (head, tail) = w.data.split_at_mut(j * w.rows);
                let wi = &head[i * w.rows..][..w.rows];
                tail[..w.rows].iter_mut().zip(wi).for_each(|(x, y)| *x -= c * y);
                if let Some(p) = &mut p {
                    let (head, tail) = p.data.split_at_mut(j * p.rows);
                    let pi = &head[i * p.rows..][..p.rows];
                    tail[..p.rows].iter_mut().zip(pi).for_each(|(x, y)| *x -= c * y);
                }
                if pass < 2 {
                    r.set(i, j, r.at(i, j) + c);
                }
            }
        }
        let norm = match &p { Some(p) => dot(w.col(j), p.col(j)).max(0.0).sqrt(), None => dot(w.col(j), w.col(j)).sqrt() };
        if norm > 1e-10 * scale && norm > 0.0 {
            r.set(j, j, norm);
            let n = w.rows;
            w.data[j * n..][..n].iter_mut().for_each(|x| *x /= norm);
            if let Some(p) = &mut p {
                p.data[j * n..][..n].iter_mut().for_each(|x| *x /= norm);
            }
            continue;
        }
        // Replace: a random direction, orthogonalized against the earlier ones, with coefficient zero.
        let n = w.rows;
        let mut rnd: Vec<f64> = (0..n).map(|_| { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17; (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5 }).collect();
        let mut wr = match (&p, m) {
            (Some(_), Some(m)) => { let mut o = vec![0.0; n]; m.apply(&rnd, &mut o).expect("apply"); o }
            _ => rnd.clone(),
        };
        for _ in 0..2 {
            for i in 0..j {
                let c = match &p { Some(_) => dot(w.col(i), &rnd), None => dot(w.col(i), &wr) };
                wr.iter_mut().zip(w.col(i)).for_each(|(x, y)| *x -= c * y);
                if let Some(p) = &p {
                    rnd.iter_mut().zip(p.col(i)).for_each(|(x, y)| *x -= c * y);
                }
            }
        }
        let nn = match &p { Some(_) => dot(&wr, &rnd).max(0.0).sqrt(), None => dot(&wr, &wr).sqrt() };
        w.data[j * n..][..n].iter_mut().zip(&wr).for_each(|(x, y)| *x = y / nn);
        if let Some(p) = &mut p {
            p.data[j * n..][..n].iter_mut().zip(&rnd).for_each(|(x, y)| *x = y / nn);
        }
        r.set(j, j, 0.0);
        REPLACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    (w, p, r.t())
}

static REPLACED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn apply_cols(op: &dyn Operator, x: &Tall, adjoint: bool) -> Tall {
    let rout = if adjoint { op.ncols() } else { op.nrows() };
    let mut out = Tall::zeros(rout, x.s);
    for j in 0..x.s {
        let o = &mut out.data[j * rout..][..rout];
        if adjoint { op.apply_adjoint(x.col(j), o) } else { op.apply(x.col(j), o) }.expect("apply");
    }
    out
}

/// Per column `‖Âᵀrⱼ‖_M / ‖Âᵀbⱼ‖_M` after each iteration, and the recomputed final value.
fn run(a: &dyn Operator, m: &dyn Operator, bs: &[&[f64]], iters: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let s = bs.len();
    let (nobs, n) = (a.nrows(), a.ncols());
    let b = Tall { rows: nobs, s, data: bs.concat() };
    let mut seed = 0x9E3779B97F4A7C15u64;
    let (mut u, _, l1) = orthonormalize(b.clone(), None, None, &mut seed);
    let beta1 = l1.t();
    let p0 = apply_cols(a, &u, true);
    let (mut v, pt, mut alpha) = orthonormalize(apply_cols(m, &p0, false), Some(p0), Some(m), &mut seed);
    let mut pt = pt.unwrap();
    let mut alpha_bar = alpha.clone();
    let mut zeta_bar = alpha.t().mul(&beta1);
    let norms0: Vec<f64> = (0..s).map(|j| zeta_bar.col_norm(j)).collect();
    let mut gbar_prev = Small::eye(2 * s);
    let mut theta = Small::zeros(s, s);
    let (mut h, mut hbar, mut x) = (Tall::zeros(n, s), Tall::zeros(n, s), Tall::zeros(n, s));
    let mut traj = vec![vec![1.0]; s];
    for _ in 0..iters {
        let mut q = apply_cols(a, &v, false);
        q.add(-1.0, &u.mul(&alpha));
        let (u_next, _, l) = orthonormalize(q, None, None, &mut seed);
        let beta = l.t();
        let mut p = apply_cols(a, &u_next, true);
        p.add(-1.0, &pt.mul(&beta.t()));
        let w = apply_cols(m, &p, false);
        let (v_next, pt_next, alpha_next) = orthonormalize(w, Some(p), Some(m), &mut seed);
        let g = Small::stack(&alpha_bar, &beta).qr_left();
        let rho = g.mul(&Small::stack(&alpha_bar, &beta)).rows(0, s);
        let next = g.mul(&Small::stack(&Small::zeros(s, s), &alpha_next));
        let theta_next = next.rows(0, s);
        alpha_bar = next.rows(s, s);
        let shifted = gbar_prev.mul(&Small::stack(&Small::zeros(s, s), &rho.t()));
        let theta_bar = shifted.rows(0, s);
        let m2 = Small::stack(&shifted.rows(s, s), &theta_next.t());
        let gbar = m2.qr_left();
        let rho_bar = gbar.mul(&m2).rows(0, s);
        let z = gbar.mul(&Small::stack(&zeta_bar, &Small::zeros(s, s)));
        let zeta = z.rows(0, s);
        zeta_bar = z.rows(s, s);
        let mut hv = v.clone();
        hv.add(-1.0, &h.mul(&theta));
        h = hv.mul(&rho.inv());
        let mut hh = h.clone();
        hh.add(-1.0, &hbar.mul(&theta_bar));
        hbar = hh.mul(&rho_bar.inv());
        x.add(1.0, &hbar.mul(&zeta));
        let mut done = true;
        for j in 0..s {
            let e = zeta_bar.col_norm(j) / norms0[j];
            done &= e <= 1e-13;
            traj[j].push(e);
        }
        (u, v, pt, alpha, theta, gbar_prev) = (u_next, v_next, pt_next.unwrap(), alpha_next, theta_next, gbar);
        if done {
            break;
        }
    }
    let mut r = b;
    r.add(-1.0, &apply_cols(a, &x, false));
    let z = apply_cols(a, &r, true);
    let w = apply_cols(m, &z, false);
    let g = z.gram(&w);
    (traj, (0..s).map(|j| g.at(j, j).max(0.0).sqrt() / norms0[j]).collect())
}

fn first_below(v: &[f64], eps: f64) -> Option<usize> {
    v.iter().position(|&x| x <= eps)
}

/// Prints, per column, iterations to each tolerance alone and in the block, and column-iteration totals.
pub(crate) fn report(a: &dyn Operator, m: &dyn Operator, bs: &[&[f64]]) {
    let iters: usize = std::env::var("BLOCK_PROBE_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let eps = [1e-6, 1e-8, 1e-10];
    REPLACED.store(0, std::sync::atomic::Ordering::Relaxed);
    let (block, rec) = run(a, m, bs, iters);
    let replaced = REPLACED.load(std::sync::atomic::Ordering::Relaxed);
    let singles: Vec<_> = bs.iter().map(|b| run(a, m, &[*b], iters)).collect();
    let show = |o: Option<usize>| o.map_or("-".to_string(), |i| i.to_string());
    eprintln!("block probe: n={} s={} replaced directions {replaced} (single/block per tol {:?})", a.ncols(), bs.len(), eps);
    for j in 0..bs.len() {
        let cells: Vec<String> = eps.iter().map(|&e| format!("{}/{}", show(first_below(&singles[j].0[0], e)), show(first_below(&block[j], e)))).collect();
        eprintln!("  col {j}: {}  recomputed single {:.0e} block {:.0e}", cells.join("  "), singles[j].1[0], rec[j]);
    }
    for &e in &eps {
        let single: Option<usize> = singles.iter().map(|s| first_below(&s.0[0], e)).sum();
        let per: Option<Vec<usize>> = block.iter().map(|t| first_below(t, e)).collect();
        if let (Some(single), Some(per)) = (single, per) {
            let max = *per.iter().max().unwrap() * bs.len();
            let frozen: usize = per.iter().sum();
            eprintln!("  tol {e:.0e}: column-its single {single}, block to max {max} ({:.2}x), frozen {frozen} ({:.2}x)", single as f64 / max as f64, single as f64 / frozen as f64);
        } else {
            eprintln!("  tol {e:.0e}: not reached by all columns within {iters}");
        }
    }
}

/// Best-of-`reps` seconds for `f`.
fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps).map(|_| { let t = std::time::Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min)
}

/// Per-kernel ms per column for `s` columns: columns one after another (each apply uses the pool) vs concurrently.
pub(crate) fn kernel_profile(a: &dyn Operator, inner: &dyn Operator, map: &dyn Operator, bs: &[&[f64]]) {
    let s = bs.len();
    let (m, n) = (a.nrows(), a.ncols());
    let mut g = vec![vec![0.0; n]; s];
    bs.iter().zip(&mut g).for_each(|(b, g)| a.apply_adjoint(b, g).unwrap());
    let mut ym = vec![vec![0.0; m]; s];
    let mut yn = vec![vec![0.0; n]; s];
    let reps = 5;
    let per = |t: f64| 1e3 * t / s as f64;
    let mut row = |name: &str, seq: f64, conc: f64| eprintln!("  {name:8} seq {:7.3}  concurrent {:7.3} ms/col", per(seq), per(conc));
    eprintln!("kernel profile: m={m} n={n} s={s} threads={}", rayon::current_num_threads());
    let seq = best(reps, || g.iter().zip(&mut ym).for_each(|(x, y)| a.apply(x, y).unwrap()));
    let conc = best(reps, || g.par_iter().zip(&mut ym).for_each(|(x, y)| a.apply(x, y).unwrap()));
    row("A", seq, conc);
    let seq = best(reps, || bs.iter().zip(&mut yn).for_each(|(x, y)| a.apply_adjoint(x, y).unwrap()));
    let conc = best(reps, || bs.par_iter().zip(&mut yn).for_each(|(x, y)| a.apply_adjoint(x, y).unwrap()));
    row("At", seq, conc);
    let seq = best(reps, || g.iter().zip(&mut yn).for_each(|(x, y)| inner.apply(x, y).unwrap()));
    let conc = best(reps, || g.par_iter().zip(&mut yn).for_each(|(x, y)| inner.apply(x, y).unwrap()));
    row("Schwarz", seq, conc);
    let seq = best(reps, || g.iter().zip(&mut yn).for_each(|(x, y)| map.apply(x, y).unwrap()));
    let conc = best(reps, || g.par_iter().zip(&mut yn).for_each(|(x, y)| map.apply(x, y).unwrap()));
    row("map", seq, conc);
    let mut big = Tall { rows: m, s, data: bs.concat() };
    let small = Tall { rows: n, s, data: g.concat() };
    let ortho = best(reps, || { let _ = small.gram(&small); let _ = big.gram(&big); let c = Small::eye(s); big = big.mul(&c); });
    row("blockvec", ortho, ortho);
}

// ---------- Fast block LSMR: interleaved rows, fused kernels, CholQR2 with rank-revealing fallback ----------

use crate::operator::reduced_design::BlockDesign;
use std::time::Instant;

type Mk<const K: usize> = [[f64; K]; K];

fn to_small<const K: usize>(m: &Mk<K>) -> Small {
    Small { r: K, c: K, a: m.iter().flatten().copied().collect() }
}

fn from_small<const K: usize>(s: &Small) -> Mk<K> {
    std::array::from_fn(|i| std::array::from_fn(|j| s.at(i, j)))
}

const CHUNK: usize = 4096;

fn add_outer<const K: usize>(g: &mut Mk<K>, a: &[f64; K], b: &[f64; K]) {
    for i in 0..K {
        for j in 0..K {
            g[i][j] += a[i] * b[j];
        }
    }
}

fn sum_mk<const K: usize>(mut a: Mk<K>, b: Mk<K>) -> Mk<K> {
    for i in 0..K {
        for j in 0..K {
            a[i][j] += b[i][j];
        }
    }
    a
}

/// `x[p] ← x[p]·r` (row times `K×K`).
#[inline(always)]
fn row_mul<const K: usize>(x: &[f64; K], r: &Mk<K>) -> [f64; K] {
    let mut o = [0.0; K];
    for i in 0..K {
        let xi = x[i];
        for j in 0..K {
            o[j] += xi * r[i][j];
        }
    }
    o
}

/// `q -= u·c` (optional), then `q ← q·r` (optional); returns `qᵀ q` (or `qᵀ p` for the paired `p` updated alike).
fn fused_pass<const K: usize>(
    q: &mut [[f64; K]],
    mut p: Option<&mut [[f64; K]]>,
    sub: Option<(&[[f64; K]], Option<&[[f64; K]]>, &Mk<K>)>,
    scale: Option<&Mk<K>>,
    gram: bool,
) -> Mk<K> {
    let zero = [[0.0; K]; K];
    let body = |q: &mut [[f64; K]], mut p: Option<&mut [[f64; K]]>, first: usize| {
        let mut g = zero;
        for (i, qi) in q.iter_mut().enumerate() {
            let r = first + i;
            if let Some((u, pu, c)) = sub {
                let d = row_mul(&u[r], c);
                (0..K).for_each(|j| qi[j] -= d[j]);
                if let (Some(p), Some(pu)) = (p.as_deref_mut(), pu) {
                    let d = row_mul(&pu[r], c);
                    (0..K).for_each(|j| p[i][j] -= d[j]);
                }
            }
            if let Some(s) = scale {
                *qi = row_mul(qi, s);
                if let Some(p) = p.as_deref_mut() {
                    p[i] = row_mul(&p[i], s);
                }
            }
            if gram {
                match p.as_deref() {
                    Some(p) => add_outer(&mut g, qi, &p[i]),
                    None => add_outer(&mut g, qi, qi),
                }
            }
        }
        g
    };
    match p.take() {
        Some(p) => q
            .par_chunks_mut(CHUNK)
            .zip(p.par_chunks_mut(CHUNK))
            .enumerate()
            .map(|(c, (q, p))| body(q, Some(p), c * CHUNK))
            .reduce(|| zero, sum_mk),
        None => q
            .par_chunks_mut(CHUNK)
            .enumerate()
            .map(|(c, q)| body(q, None, c * CHUNK))
            .reduce(|| zero, sum_mk),
    }
}

fn sym<const K: usize>(g: &Mk<K>) -> Small {
    let mut s = to_small(g);
    for i in 0..K {
        for j in 0..i {
            let v = 0.5 * (s.at(i, j) + s.at(j, i));
            s.set(i, j, v);
            s.set(j, i, v);
        }
    }
    s
}

fn to_tall<const K: usize>(x: &[[f64; K]]) -> Tall {
    let n = x.len();
    let mut t = Tall::zeros(n, K);
    for (i, r) in x.iter().enumerate() {
        for j in 0..K {
            t.data[j * n + i] = r[j];
        }
    }
    t
}

fn from_tall<const K: usize>(t: &Tall, x: &mut [[f64; K]]) {
    let n = t.rows;
    for (i, r) in x.iter_mut().enumerate() {
        for j in 0..K {
            r[j] = t.data[j * n + i];
        }
    }
}

/// CholQR2 of `q` (Euclidean) or of `(w = q, p)` under the metric `wᵀp`; returns `L` with `q_in = q_out Lᵀ`.
/// `sub` folds `q -= u c` (and `p -= pu c`) into the first pass; Cholesky failure falls back to rank-revealing MGS.
fn cholqr2_k<const K: usize>(
    q: &mut [[f64; K]],
    mut p: Option<&mut [[f64; K]]>,
    sub: Option<(&[[f64; K]], Option<&[[f64; K]]>, &Mk<K>)>,
    m: Option<&dyn Operator>,
    seed: &mut u64,
) -> Small {
    let mut total = Small::eye(K);
    let mut g = fused_pass(q, p.as_deref_mut(), sub, None, true);
    let adaptive = std::env::var_os("PROBE_CHOLQR2").is_none();
    for stage in 0..2 {
        match sym(&g).cholesky() {
            Some(l) => {
                let r = from_small::<K>(&l.t().inv());
                total = total.mul(&l);
                // CholQR's orthogonality error is ~κ(G)ε = κ(q)²ε: one pass suffices while κ(q) ≤ 1e3.
                let d: Vec<f64> = (0..K).map(|i| l.at(i, i)).collect();
                let ratio = d.iter().fold(0.0f64, |a, &b| a.max(b)) / d.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let last = stage == 1 || (adaptive && ratio <= 1e3);
                g = fused_pass(q, p.as_deref_mut(), None, Some(&r), !last);
                if last {
                    return total;
                }
            }
            None => {
                let (w, pp, l) = orthonormalize(to_tall(q), p.as_deref().map(to_tall), m, seed);
                from_tall(&w, q);
                if let (Some(p), Some(pp)) = (p.as_deref_mut(), pp) {
                    from_tall(&pp, p);
                }
                return total.mul(&l);
            }
        }
    }
    total
}

fn apply_concurrent<const K: usize>(op: &dyn Operator, x: &[[f64; K]], y: &mut [[f64; K]], adjoint: bool) {
    let (nin, nout) = (x.len(), y.len());
    let cols: Vec<Vec<f64>> = (0..K).into_par_iter().map(|j| {
        let xin: Vec<f64> = x.iter().map(|r| r[j]).collect();
        let mut out = vec![0.0; nout];
        if adjoint { op.apply_adjoint(&xin, &mut out) } else { op.apply(&xin, &mut out) }.expect("apply");
        let _ = nin;
        out
    }).collect();
    y.par_chunks_mut(CHUNK).enumerate().for_each(|(c, rows)| {
        for (i, r) in rows.iter_mut().enumerate() {
            for j in 0..K {
                r[j] = cols[j][c * CHUNK + i];
            }
        }
    });
}

pub(crate) struct FastRun {
    pub iterations: usize,
    pub per_column: Vec<usize>,
    pub wall: f64,
    pub secs: [f64; 4],
    pub recomputed: Vec<f64>,
    pub xs: Vec<Vec<f64>>,
    pub achieved: Vec<f64>,
}

/// Block LSMR to `‖Âᵀrⱼ‖_M ≤ tol ‖Âᵀbⱼ‖_M` for every column; `fused` uses the k-column design kernels.
fn fast<const K: usize>(
    a: &dyn Operator,
    ab: &dyn BlockDesign,
    m: &dyn Operator,
    bs: &[&[f64]],
    tols: &[f64],
    maxit: usize,
    fused: bool,
    mb: Option<BlockMap<'_>>,
    verify: bool,
) -> FastRun {
    let (nobs, n) = (a.nrows(), a.ncols());
    let t0 = Instant::now();
    let mut secs = [0.0; 4];
    let mut seed = 0x9E3779B97F4A7C15u64;
    let ap = |x: &[[f64; K]], y: &mut [[f64; K]], secs: &mut [f64; 4]| {
        let t = Instant::now();
        if fused { ab.apply_block(x.as_flattened(), y.as_flattened_mut(), K) } else { apply_concurrent(a, x, y, false) }
        secs[0] += t.elapsed().as_secs_f64();
    };
    let at = |x: &[[f64; K]], y: &mut [[f64; K]], secs: &mut [f64; 4]| {
        let t = Instant::now();
        if fused { ab.apply_adjoint_block(x.as_flattened(), y.as_flattened_mut(), K) } else { apply_concurrent(a, x, y, true) }
        secs[1] += t.elapsed().as_secs_f64();
    };
    let mp = |x: &[[f64; K]], y: &mut [[f64; K]], secs: &mut [f64; 4]| {
        let t = Instant::now();
        match mb {
            Some(f) => f(x.as_flattened(), y.as_flattened_mut(), K),
            None => apply_concurrent(m, x, y, false),
        }
        secs[2] += t.elapsed().as_secs_f64();
    };
    let interleave = || {
        let mut b = vec![[0.0; K]; nobs];
        b.par_chunks_mut(CHUNK).enumerate().for_each(|(c, rows)| {
            for (i, r) in rows.iter_mut().enumerate() {
                for j in 0..K {
                    r[j] = bs[j][c * CHUNK + i];
                }
            }
        });
        b
    };
    let mut u = interleave();
    let tv = Instant::now();
    let beta1 = cholqr2_k(&mut u, None, None, None, &mut seed).t();
    secs[3] += tv.elapsed().as_secs_f64();
    let mut p = vec![[0.0; K]; n];
    at(&u, &mut p, &mut secs);
    let mut v = vec![[0.0; K]; n];
    mp(&p, &mut v, &mut secs);
    let tv = Instant::now();
    let mut alpha = cholqr2_k(&mut v, Some(&mut p), None, Some(m), &mut seed);
    secs[3] += tv.elapsed().as_secs_f64();
    let mut pt = p;
    let mut alpha_bar = alpha.clone();
    let mut zeta_bar = alpha.t().mul(&beta1);
    let norms0: Vec<f64> = (0..K).map(|j| zeta_bar.col_norm(j)).collect();
    let mut gbar_prev = Small::eye(2 * K);
    let mut theta = Small::zeros(K, K);
    let (mut h, mut hbar, mut x) = (vec![[0.0; K]; n], vec![[0.0; K]; n], vec![[0.0; K]; n]);
    let mut q = vec![[0.0; K]; nobs];
    let (mut p, mut w) = (vec![[0.0; K]; n], vec![[0.0; K]; n]);
    let mut per_column = vec![0usize; K];
    let mut it = 0;
    while it < maxit {
        it += 1;
        ap(&v, &mut q, &mut secs);
        let tv = Instant::now();
        let am = from_small::<K>(&alpha);
        let beta = cholqr2_k(&mut q, None, Some((&u, None, &am)), None, &mut seed).t();
        std::mem::swap(&mut u, &mut q);
        secs[3] += tv.elapsed().as_secs_f64();
        at(&u, &mut p, &mut secs);
        let bt = from_small::<K>(&beta.t());
        let tv = Instant::now();
        fused_pass(&mut p, None, Some((&pt, None, &bt)), None, false);
        secs[3] += tv.elapsed().as_secs_f64();
        mp(&p, &mut w, &mut secs);
        let tv = Instant::now();
        let alpha_next = cholqr2_k(&mut w, Some(&mut p), None, Some(m), &mut seed);
        let g = Small::stack(&alpha_bar, &beta).qr_left();
        let rho = g.mul(&Small::stack(&alpha_bar, &beta)).rows(0, K);
        let next = g.mul(&Small::stack(&Small::zeros(K, K), &alpha_next));
        let theta_next = next.rows(0, K);
        alpha_bar = next.rows(K, K);
        let shifted = gbar_prev.mul(&Small::stack(&Small::zeros(K, K), &rho.t()));
        let theta_bar = shifted.rows(0, K);
        let m2 = Small::stack(&shifted.rows(K, K), &theta_next.t());
        let gbar = m2.qr_left();
        let rho_bar = gbar.mul(&m2).rows(0, K);
        let z = gbar.mul(&Small::stack(&zeta_bar, &Small::zeros(K, K)));
        let zeta = from_small::<K>(&z.rows(0, K));
        zeta_bar = z.rows(K, K);
        let (th, rinv) = (from_small::<K>(&theta), from_small::<K>(&rho.inv()));
        let (thb, rbinv) = (from_small::<K>(&theta_bar), from_small::<K>(&rho_bar.inv()));
        let vref = &v;
        h.par_chunks_mut(CHUNK)
            .zip(hbar.par_chunks_mut(CHUNK))
            .zip(x.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(c, ((h, hb), x))| {
                for i in 0..h.len() {
                    let r = c * CHUNK + i;
                    let d = row_mul(&h[i], &th);
                    let mut hv = vref[r];
                    (0..K).for_each(|j| hv[j] -= d[j]);
                    h[i] = row_mul(&hv, &rinv);
                    let d = row_mul(&hb[i], &thb);
                    let mut hh = h[i];
                    (0..K).for_each(|j| hh[j] -= d[j]);
                    hb[i] = row_mul(&hh, &rbinv);
                    let d = row_mul(&hb[i], &zeta);
                    (0..K).for_each(|j| x[i][j] += d[j]);
                }
            });
        secs[3] += tv.elapsed().as_secs_f64();
        let mut done = true;
        for j in 0..K {
            if per_column[j] == 0 && zeta_bar.col_norm(j) <= tols[j] * norms0[j] {
                per_column[j] = it;
            }
            done &= per_column[j] > 0;
        }
        std::mem::swap(&mut v, &mut w);
        std::mem::swap(&mut pt, &mut p);
        (alpha, theta, gbar_prev) = (alpha_next, theta_next, gbar);
        if done {
            break;
        }
    }
    let achieved: Vec<f64> = (0..K).map(|j| zeta_bar.col_norm(j) / norms0[j]).collect();
    drop((u, q, v, w, p, pt, h, hbar));
    let xs: Vec<Vec<f64>> = (0..K).into_par_iter().map(|j| x.iter().map(|r| r[j]).collect()).collect();
    let wall = t0.elapsed().as_secs_f64();
    if !verify {
        return FastRun { iterations: it, per_column, wall, secs, recomputed: Vec::new(), xs, achieved };
    }
    let b = interleave();
    let mut r = vec![[0.0; K]; nobs];
    ab.apply_block(x.as_flattened(), r.as_flattened_mut(), K);
    r.iter_mut().zip(&b).for_each(|(r, b)| (0..K).for_each(|j| r[j] = b[j] - r[j]));
    let mut z = vec![[0.0; K]; n];
    ab.apply_adjoint_block(r.as_flattened(), z.as_flattened_mut(), K);
    let mut mz = vec![[0.0; K]; n];
    apply_concurrent(m, &z, &mut mz, false);
    let recomputed = (0..K)
        .map(|j| z.iter().zip(&mz).map(|(a, b)| a[j] * b[j]).sum::<f64>().max(0.0).sqrt() / norms0[j])
        .collect();
    FastRun { iterations: it, per_column, wall, secs, recomputed, xs, achieved }
}

pub(crate) type BlockMap<'a> = &'a (dyn Fn(&[f64], &mut [f64], usize) + Sync);

macro_rules! dispatch_fast {
    ($k:expr, $($arg:expr),*) => {
        match $k {
            1 => fast::<1>($($arg),*), 2 => fast::<2>($($arg),*), 3 => fast::<3>($($arg),*),
            4 => fast::<4>($($arg),*), 5 => fast::<5>($($arg),*), 6 => fast::<6>($($arg),*),
            7 => fast::<7>($($arg),*), 8 => fast::<8>($($arg),*), 12 => fast::<12>($($arg),*),
            16 => fast::<16>($($arg),*),
            _ => panic!("k = {} unsupported", $k),
        }
    };
}

/// Wall clock: block (fused and unfused design kernels) vs the same code at K=1, columns concurrent.
/// ms per column: block vs concurrent per-column, for the inner Schwarz map and the whole map.
pub(crate) fn m_timing(inner: &dyn Operator, ib: BlockMap<'_>, m: &dyn Operator, mb: BlockMap<'_>, s: usize) {
    let n = m.ncols();
    let x: Vec<f64> = (0..n * s).map(|i| ((i * 7919) % 1000) as f64 / 1000.0 - 0.5).collect();
    let cols: Vec<Vec<f64>> = (0..s).map(|j| (0..n).map(|i| x[i * s + j]).collect()).collect();
    let mut y = vec![0.0; n * s];
    let mut yc = vec![vec![0.0; n]; s];
    let per = |t: f64| 1e3 * t / s as f64;
    let ib_t = best(5, || ib(&x, &mut y, s));
    let in_t = best(5, || cols.par_iter().zip(&mut yc).for_each(|(x, y)| inner.apply(x, y).unwrap()));
    crate::operator::reduced_coarse::CTIME.iter().for_each(|c| c.store(0, std::sync::atomic::Ordering::Relaxed));
    let mb_t = best(5, || mb(&x, &mut y, s));
    let parts: Vec<f64> = crate::operator::reduced_coarse::CTIME.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-6 / 5.0 / s as f64).collect();
    eprintln!("  coarse block parts ms/col (Z, E-solve, SZ, hubs, inner): {:.3} {:.3} {:.3} {:.3} {:.3}", parts[0], parts[1], parts[2], parts[3], parts[4]);
    let m_t = best(5, || cols.par_iter().zip(&mut yc).for_each(|(x, y)| m.apply(x, y).unwrap()));
    eprintln!("  M timing ms/col: inner block {:.3} concurrent {:.3} | map block {:.3} concurrent {:.3}", per(ib_t), per(in_t), per(mb_t), per(m_t));
}

/// `‖Âᵀ(b − Âx)‖_M / ‖Âᵀb‖_M`, the block run's stopping metric.
fn metric_rel(a: &dyn Operator, m: &dyn Operator, b: &[f64], x: &[f64]) -> f64 {
    let (nobs, n) = (a.nrows(), a.ncols());
    let mut ax = vec![0.0; nobs];
    a.apply(x, &mut ax).unwrap();
    let r: Vec<f64> = b.iter().zip(&ax).map(|(b, ax)| b - ax).collect();
    let mnorm = |v: &[f64]| {
        let mut z = vec![0.0; n];
        a.apply_adjoint(v, &mut z).unwrap();
        let mut mz = vec![0.0; n];
        m.apply(&z, &mut mz).unwrap();
        z.iter().zip(&mz).map(|(a, b)| a * b).sum::<f64>().max(0.0).sqrt()
    };
    mnorm(&r) / mnorm(b)
}

/// Production `mlsmr` per column (concurrent, cold) vs block LSMR stopped at the accuracy production reached.
pub(crate) fn wall_report(
    a: &dyn Operator,
    ab: &dyn BlockDesign,
    m: &dyn Operator,
    mb: BlockMap<'_>,
    bs: &[&[f64]],
    refs: &[Option<f64>],
    tol: f64,
    local_size: Option<usize>,
) {
    let maxit = 5000;
    let s = bs.len();
    if std::env::var_os("BLOCK_ONLY").is_some() {
        let r = dispatch_fast!(s, a, ab, m, bs, &vec![tol; s], maxit, true, Some(mb), true);
        eprintln!("  block only: {:.1} ms, {} its", 1e3 * r.wall, r.iterations);
        return;
    }
    let rounds: usize = std::env::var("BLOCK_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    {
        let n = m.ncols();
        let x: Vec<f64> = (0..n * s).map(|i| ((i * 7919) % 1000) as f64 / 1000.0 - 0.5).collect();
        let mut yb = vec![0.0; n * s];
        mb(&x, &mut yb, s);
        let mut worst = 0.0f64;
        for j in 0..s {
            let xc: Vec<f64> = (0..n).map(|i| x[i * s + j]).collect();
            let mut yc = vec![0.0; n];
            m.apply(&xc, &mut yc).unwrap();
            let num: f64 = (0..n).map(|i| (yc[i] - yb[i * s + j]).powi(2)).sum::<f64>().sqrt();
            let den: f64 = yc.iter().map(|v| v * v).sum::<f64>().sqrt();
            worst = worst.max(num / den);
        }
        eprintln!("  fused M check: max relative difference vs per-column apply {worst:.1e}");
    }
    let ms = |t: f64| 1e3 * t;
    let mut best_prod = f64::INFINITY;
    let mut prod: Vec<schwarz_precond::LsmrResult> = Vec::new();
    let mut fused: Option<FastRun> = None;
    let mut fused_d: Option<FastRun> = None;
    let keep = |slot: &mut Option<FastRun>, r: FastRun| {
        if slot.as_ref().is_none_or(|o| r.wall < o.wall) {
            *slot = Some(r);
        }
    };
    let mut tols = Vec::new();
    for round in 0..rounds {
        let t = Instant::now();
        prod = bs
            .par_iter()
            .zip(refs)
            .map(|(b, r)| {
                let options = schwarz_precond::MlsmrOptions { reference_norm: *r, local_size, ..Default::default() };
                schwarz_precond::mlsmr(a, b, m, tol, maxit, options).expect("mlsmr")
            })
            .collect();
        best_prod = best_prod.min(t.elapsed().as_secs_f64());
        if round == 0 {
            tols = bs.iter().zip(&prod).map(|(b, r)| metric_rel(a, m, b, &r.x)).collect();
        }
        keep(&mut fused_d, dispatch_fast!(s, a, ab, m, bs, &tols, maxit, true, None, true));
        keep(&mut fused, dispatch_fast!(s, a, ab, m, bs, &tols, maxit, true, Some(mb), true));
    }
    let (fused, fused_d) = (fused.unwrap(), fused_d.unwrap());
    let its: Vec<usize> = prod.iter().map(|r| r.iterations).collect();
    let sum: usize = its.iter().sum();
    eprintln!("wall report: s={s} tol={tol:.0e} threads={}", rayon::current_num_threads());
    eprintln!("  production mlsmr concurrent: {:.1} ms, iterations {its:?} (sum {sum}), {:.3} ms/col-it·wall", ms(best_prod), ms(best_prod) / sum as f64);
    eprintln!("  matched tolerances (block metric) {:?}", tols.iter().map(|t| format!("{t:.1e}")).collect::<Vec<_>>());
    for (name, r) in [("block fused A+M", &fused), ("block fused A, concurrent M", &fused_d)] {
        let worst = r.recomputed.iter().zip(&tols).map(|(a, t)| a / t).fold(0.0f64, f64::max);
        eprintln!(
            "  {name}: {:.1} ms ({:.2}x vs production), {} its ({} col-its, {:.3} ms/col-it), A {:.1} At {:.1} M {:.1} vec {:.1} ms, recomputed/target max {worst:.2}",
            ms(r.wall), best_prod / r.wall, r.iterations, r.iterations * s, ms(r.wall) / (r.iterations * s) as f64,
            ms(r.secs[0]), ms(r.secs[1]), ms(r.secs[2]), ms(r.secs[3]),
        );
    }
}

/// The block solve itself: one result per column, nothing kept for verification.
pub(crate) fn solve(
    a: &dyn Operator,
    ab: &dyn BlockDesign,
    m: &dyn Operator,
    mb: BlockMap<'_>,
    bs: &[&[f64]],
    tol: f64,
    maxit: usize,
) -> Vec<(schwarz_precond::LsmrResult, f64)> {
    let r = dispatch_fast!(bs.len(), a, ab, m, bs, &vec![tol; bs.len()], maxit, true, Some(mb), false);
    eprintln!("  block solve: {:.1} ms, {} its", 1e3 * r.wall, r.iterations);
    r.xs.into_iter()
        .zip(r.per_column)
        .zip(r.achieved)
        .map(|((x, its), achieved)| {
            let converged = its > 0;
            let result = schwarz_precond::LsmrResult {
                x,
                converged,
                iterations: if converged { its } else { r.iterations },
                residual_norm: f64::NAN,
                normal_eq_residual: achieved,
                stop_reason: if converged {
                    schwarz_precond::LsmrStopReason::NormalEquationTolerance
                } else {
                    schwarz_precond::LsmrStopReason::MaxIterations
                },
                true_residual: None,
            };
            (result, r.wall)
        })
        .collect()
}
