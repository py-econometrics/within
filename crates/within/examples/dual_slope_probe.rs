use std::{collections::HashMap, error::Error, time::Instant};

use schwarz_precond::{mlsmr, MlsmrOptions, Operator, SolveError};
use within::{Effect, LsmrOptions, PreconditionerConfig, Solver};

struct Panel {
    worker: Vec<u32>,
    firm: Vec<u32>,
    worker_slope: Vec<f64>,
    firm_slope: Vec<f64>,
    y: Vec<f64>,
}

impl Panel {
    fn read(path: &str) -> Result<Self, Box<dyn Error>> {
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines();
        let names: HashMap<_, _> = lines
            .next()
            .ok_or("missing CSV header")?
            .split(',')
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();
        let mut p = Self {
            worker: vec![],
            firm: vec![],
            worker_slope: vec![],
            firm_slope: vec![],
            y: vec![],
        };
        for line in lines {
            let row: Vec<&str> = line.split(',').collect();
            p.worker.push(row[names["worker"]].parse()?);
            p.firm.push(row[names["firm"]].parse()?);
            p.worker_slope.push(row[names["worker_slope"]].parse()?);
            p.firm_slope.push(row[names["firm_slope"]].parse()?);
            p.y.push(row[names["y"]].parse()?);
        }
        assert!(!p.y.is_empty() && p.worker.is_sorted());
        Ok(p)
    }
}

struct Term {
    levels: Vec<u32>,
    rows: Vec<[f64; 2]>,
    n: usize,
}

impl Term {
    fn new(levels: &[u32], slope: &[f64]) -> Self {
        let n = *levels.iter().max().unwrap() as usize + 1;
        let mut count = vec![0.0_f64; n];
        let mut mean = vec![0.0; n];
        for (&l, &z) in levels.iter().zip(slope) {
            count[l as usize] += 1.0;
            mean[l as usize] += z;
        }
        for (m, &c) in mean.iter_mut().zip(&count) {
            *m /= c.max(1.0);
        }
        let mut norm = vec![0.0; n];
        for (&l, &z) in levels.iter().zip(slope) {
            norm[l as usize] += (z - mean[l as usize]).powi(2);
        }
        let rows = levels
            .iter()
            .zip(slope)
            .map(|(&l, &z)| {
                let l = l as usize;
                [
                    1.0 / count[l].sqrt(),
                    if norm[l] > 0.0 {
                        (z - mean[l]) / norm[l].sqrt()
                    } else {
                        0.0
                    },
                ]
            })
            .collect();
        Self {
            levels: levels.to_vec(),
            rows,
            n,
        }
    }

    fn gather(&self, x: &[f64], y: &mut [f64]) {
        for ((v, &l), row) in y.iter_mut().zip(&self.levels).zip(&self.rows) {
            *v = row[0] * x[2 * l as usize] + row[1] * x[2 * l as usize + 1];
        }
    }

    fn scatter(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        for ((&v, &l), row) in x.iter().zip(&self.levels).zip(&self.rows) {
            y[2 * l as usize] += row[0] * v;
            y[2 * l as usize + 1] += row[1] * v;
        }
    }
}

struct Reduced {
    worker: Term,
    firm: Term,
    starts: Vec<usize>,
}

impl Reduced {
    fn new(p: &Panel) -> Self {
        let worker = Term::new(&p.worker, &p.worker_slope);
        let firm = Term::new(&p.firm, &p.firm_slope);
        let mut starts = vec![0];
        starts.extend((1..p.y.len()).filter(|&i| p.worker[i] != p.worker[i - 1]));
        starts.push(p.y.len());
        Self {
            worker,
            firm,
            starts,
        }
    }

    fn project(&self, y: &mut [f64]) {
        for bounds in self.starts.windows(2) {
            let range = bounds[0]..bounds[1];
            let mut fit = [0.0; 2];
            for i in range.clone() {
                for (k, f) in fit.iter_mut().enumerate() {
                    *f += self.worker.rows[i][k] * y[i];
                }
            }
            for i in range {
                y[i] -= self.worker.rows[i][0] * fit[0] + self.worker.rows[i][1] * fit[1];
            }
        }
    }

    fn blocks(&self) -> Vec<[f64; 3]> {
        let mut blocks = vec![[0.0; 3]; self.firm.n];
        let mut firms = Vec::new();
        for bounds in self.starts.windows(2) {
            let range = bounds[0]..bounds[1];
            firms.clear();
            firms.extend_from_slice(&self.firm.levels[range.clone()]);
            firms.sort_unstable();
            firms.dedup();
            for &firm in &firms {
                let mut moments = [[0.0; 2]; 2];
                for i in range.clone().filter(|&i| self.firm.levels[i] == firm) {
                    for (k, row) in moments.iter_mut().enumerate() {
                        for (j, m) in row.iter_mut().enumerate() {
                            *m += self.worker.rows[i][k] * self.firm.rows[i][j];
                        }
                    }
                }
                // Squared projected columns avoid cancellation for nonmovers and short spells.
                for i in range.clone() {
                    let mut r = [0.0; 2];
                    for (j, v) in r.iter_mut().enumerate() {
                        *v = if self.firm.levels[i] == firm {
                            self.firm.rows[i][j]
                        } else {
                            0.0
                        };
                        for (k, row) in moments.iter().enumerate() {
                            *v -= self.worker.rows[i][k] * row[j];
                        }
                    }
                    let block = &mut blocks[firm as usize];
                    block[0] += r[0] * r[0];
                    block[1] += r[0] * r[1];
                    block[2] += r[1] * r[1];
                }
            }
        }
        blocks
    }
}

impl Operator for Reduced {
    fn nrows(&self) -> usize {
        self.firm.rows.len()
    }
    fn ncols(&self) -> usize {
        2 * self.firm.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.firm.gather(x, y);
        self.project(y);
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let mut r = x.to_vec();
        self.project(&mut r);
        self.firm.scatter(&r, y);
        Ok(())
    }
}

struct Blocks(Vec<[f64; 3]>);

impl Blocks {
    fn inverse(blocks: &[[f64; 3]], coupled: bool) -> Self {
        let inverse = |v: f64| if v > 1e-12 { 1.0 / v } else { 1.0 };
        Self(
            blocks
                .iter()
                .map(|&[a, b, c]| {
                    if !coupled {
                        return [inverse(a), 0.0, inverse(c)];
                    }
                    let high = 0.5 * (a + c + (a - c).hypot(2.0 * b));
                    let low = if high > 0.0 {
                        (a * c - b * b).max(0.0) / high
                    } else {
                        0.0
                    };
                    let angle = 0.5 * (2.0 * b).atan2(a - c);
                    let (s, t) = angle.sin_cos();
                    let (h, l) = (inverse(high), inverse(low));
                    [
                        h * t * t + l * s * s,
                        (h - l) * t * s,
                        h * s * s + l * t * t,
                    ]
                })
                .collect(),
        )
    }
}

impl Operator for Blocks {
    fn nrows(&self) -> usize {
        2 * self.0.len()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        for ((x, y), &[a, b, c]) in x.chunks_exact(2).zip(y.chunks_exact_mut(2)).zip(&self.0) {
            y[0] = a * x[0] + b * x[1];
            y[1] = b * x[0] + c * x[1];
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

struct Coarse {
    labels: Vec<usize>,
    count: usize,
    active: usize,
    q: Vec<f64>,
    sq: Vec<f64>,
}

impl Coarse {
    fn new(a: &Reduced, slope: &[f64]) -> Self {
        let nw = a.worker.n;
        let mut parent: Vec<usize> = (0..nw + a.firm.n).collect();
        for (&w, &f) in a.worker.levels.iter().zip(&a.firm.levels) {
            let (i, j) = (
                root(&mut parent, w as usize),
                root(&mut parent, nw + f as usize),
            );
            parent[i] = j;
        }
        let labels: Vec<usize> = (0..a.firm.n).map(|f| root(&mut parent, nw + f)).collect();
        let count = parent.len();
        let mut q = vec![0.0; a.ncols()];
        a.firm.scatter(slope, &mut q);
        let mut constant = vec![0.0; a.ncols()];
        a.firm.scatter(&vec![1.0; a.nrows()], &mut constant);
        let mut sums = vec![[0.0; 2]; count];
        for i in 0..q.len() {
            sums[labels[i / 2]][0] += q[i] * constant[i];
            sums[labels[i / 2]][1] += constant[i] * constant[i];
        }
        for i in 0..q.len() {
            let [s, d] = sums[labels[i / 2]];
            q[i] -= constant[i] * s / d.max(1e-30);
        }
        let mut image = vec![0.0; a.nrows()];
        a.apply(&q, &mut image).unwrap();
        sums.fill([0.0; 2]);
        for (&f, &v) in a.firm.levels.iter().zip(&image) {
            sums[labels[f as usize]][0] += v * v;
        }
        for i in 0..q.len() {
            sums[labels[i / 2]][1] += q[i] * q[i];
        }
        let active = sums.iter().filter(|[e, n]| *e > 1e-20 * n.max(1.0)).count();
        for i in 0..q.len() {
            let [e, n] = sums[labels[i / 2]];
            q[i] *= if e > 1e-20 * n.max(1.0) {
                1.0 / e.sqrt()
            } else {
                0.0
            };
        }
        a.apply(&q, &mut image).unwrap();
        let mut sq = vec![0.0; a.ncols()];
        a.apply_adjoint(&image, &mut sq).unwrap();
        Self {
            labels,
            count,
            active,
            q,
            sq,
        }
    }

    fn restrict(&self, x: &[f64], basis: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.count];
        for i in 0..x.len() {
            out[self.labels[i / 2]] += basis[i] * x[i];
        }
        out
    }
}

struct Balanced<'a> {
    blocks: &'a Blocks,
    coarse: &'a Coarse,
}

impl Operator for Balanced<'_> {
    fn nrows(&self) -> usize {
        self.blocks.nrows()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let c = self.coarse;
        let alpha = c.restrict(x, &c.q);
        let r: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v - c.sq[i] * alpha[c.labels[i / 2]])
            .collect();
        self.blocks.apply(&r, y)?;
        let beta = c.restrict(y, &c.sq);
        for (i, v) in y.iter_mut().enumerate() {
            let l = c.labels[i / 2];
            *v += c.q[i] * (alpha[l] - beta[l]);
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

struct Full<'a>(&'a Reduced);

impl Operator for Full<'_> {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    fn ncols(&self) -> usize {
        2 * self.0.worker.n + self.0.ncols()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let a = self.0;
        a.worker.gather(x, y);
        let x = &x[2 * a.worker.n..];
        for ((v, &f), row) in y.iter_mut().zip(&a.firm.levels).zip(&a.firm.rows) {
            *v += row[0] * x[2 * f as usize] + row[1] * x[2 * f as usize + 1];
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let (worker, firm) = y.split_at_mut(2 * self.0.worker.n);
        self.0.worker.scatter(x, worker);
        self.0.firm.scatter(x, firm);
        Ok(())
    }
}

struct Coupling {
    edges: Vec<(usize, usize, [[f64; 2]; 2])>,
}

impl Coupling {
    fn new(a: &Reduced) -> Self {
        let mut edges = Vec::new();
        let mut moments: HashMap<usize, [[f64; 2]; 2]> = HashMap::new();
        for bounds in a.starts.windows(2) {
            moments.clear();
            for i in bounds[0]..bounds[1] {
                let block = moments
                    .entry(a.firm.levels[i] as usize)
                    .or_insert([[0.0; 2]; 2]);
                for (k, row) in block.iter_mut().enumerate() {
                    for (j, v) in row.iter_mut().enumerate() {
                        *v += a.worker.rows[i][k] * a.firm.rows[i][j];
                    }
                }
            }
            let w = a.worker.levels[bounds[0]] as usize;
            edges.extend(moments.iter().map(|(&f, &block)| (w, f, block)));
        }
        edges.sort_unstable_by_key(|&(w, f, _)| (w, f));
        Self { edges }
    }

    fn subtract(&self, x: &[f64], y: &mut [f64], transpose: bool) {
        for &(w, f, block) in &self.edges {
            for (k, row) in block.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    if transpose {
                        y[2 * f + j] -= v * x[2 * w + k];
                    } else {
                        y[2 * w + k] -= v * x[2 * f + j];
                    }
                }
            }
        }
    }
}

struct PairBasis<'a> {
    base: &'a within::Preconditioner,
    indices: Vec<usize>,
    scales: Vec<f64>,
}

impl<'a> PairBasis<'a> {
    fn new(base: &'a within::Preconditioner, a: &Reduced) -> Self {
        let mut indices = Vec::new();
        let mut scales = Vec::new();
        let mut offset = 0;
        for term in [&a.worker, &a.firm] {
            let mut counts = vec![0.0_f64; term.n];
            for &l in &term.levels {
                counts[l as usize] += 1.0;
            }
            for (l, &count) in counts.iter().enumerate() {
                indices.extend([offset + l, offset + term.n + l]);
                scales.extend([count.sqrt(), 1.0]);
            }
            offset += 2 * term.n;
        }
        Self {
            base,
            indices,
            scales,
        }
    }
}

impl Operator for PairBasis<'_> {
    fn nrows(&self) -> usize {
        self.indices.len()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        let mut rhs = vec![0.0; x.len()];
        let mut out = vec![0.0; x.len()];
        for (i, &j) in self.indices.iter().enumerate() {
            rhs[j] = self.scales[i] * x[i];
        }
        self.base.apply(&rhs, &mut out)?;
        for (i, &j) in self.indices.iter().enumerate() {
            y[i] = self.scales[i] * out[j];
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

struct AugmentedPairs<'a> {
    base: &'a PairBasis<'a>,
    coupling: &'a Coupling,
    correction: &'a dyn Operator,
}

impl Operator for AugmentedPairs<'_> {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.base.apply(x, y)?;
        let nw = x.len() - self.correction.nrows();
        let (worker, firm) = x.split_at(nw);
        let mut rhs = firm.to_vec();
        self.coupling.subtract(worker, &mut rhs, true);
        let mut correction = vec![0.0; rhs.len()];
        self.correction.apply(&rhs, &mut correction)?;
        let (worker, firm) = y.split_at_mut(nw);
        self.coupling.subtract(&correction, worker, false);
        for (v, z) in firm.iter_mut().zip(correction) {
            *v += z;
        }
        Ok(())
    }
    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.apply(x, y)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let tol = std::env::var("PROBE_TOL")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(1e-12);
    println!("case,method,setup_seconds,solve_seconds,iterations,converged,relative_gradient,fit_difference,coarse_rank");
    for path in std::env::args().skip(1) {
        let p = Panel::read(&path)?;
        let start = Instant::now();
        let effects = vec![
            Effect::new(&p.worker, true, [&p.worker_slope[..]])?,
            Effect::new(&p.firm, true, [&p.firm_slope[..]])?,
        ];
        let solver = Solver::new(
            effects,
            None,
            PreconditionerConfig::Additive {
                local_solver: Default::default(),
                reduction: Default::default(),
            },
        )?;
        let setup = start.elapsed().as_secs_f64();
        let options = LsmrOptions {
            tol,
            maxiter: 20_000,
            local_size: Some(10),
        };
        let start = Instant::now();
        let baseline = solver.solve(&p.y, &options)?;
        let elapsed = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let a = Reduced::new(&p);
        let mut rhs = p.y.clone();
        a.project(&mut rhs);
        let raw = a.blocks();
        let diagonal = Blocks::inverse(&raw, false);
        let blocks = Blocks::inverse(&raw, true);
        let setup_blocks = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let coarse = Coarse::new(&a, &p.worker_slope);
        let setup_coarse = start.elapsed().as_secs_f64();
        let balanced = Balanced {
            blocks: &blocks,
            coarse: &coarse,
        };
        if std::env::var_os("PROBE_AUGMENT").is_some() {
            let start = Instant::now();
            let coupling = Coupling::new(&a);
            let pairs = PairBasis::new(solver.preconditioner().unwrap(), &a);
            let setup_coupling = start.elapsed().as_secs_f64();
            let augmented = AugmentedPairs {
                base: &pairs,
                coupling: &coupling,
                correction: &blocks,
            };
            let augmented_coarse = AugmentedPairs {
                base: &pairs,
                coupling: &coupling,
                correction: &balanced,
            };
            let full = Full(&a);
            let mut initial = vec![0.0; full.ncols()];
            full.apply_adjoint(&p.y, &mut initial)?;
            for (name, map, extra) in [
                ("pairs_control", &pairs as &dyn Operator, 0.0),
                (
                    "pairs_plus_blocks",
                    &augmented as &dyn Operator,
                    setup_blocks + setup_coupling,
                ),
                (
                    "pairs_plus_blocks_coarse",
                    &augmented_coarse as &dyn Operator,
                    setup_blocks + setup_coupling + setup_coarse,
                ),
            ] {
                let start = Instant::now();
                let result = mlsmr(
                    &full,
                    &p.y,
                    map,
                    tol,
                    20_000,
                    MlsmrOptions {
                        local_size: Some(10),
                        ..Default::default()
                    },
                )?;
                let elapsed = start.elapsed().as_secs_f64();
                let mut r = vec![0.0; full.nrows()];
                full.apply(&result.x, &mut r)?;
                for (v, &b) in r.iter_mut().zip(&p.y) {
                    *v = b - *v;
                }
                let mut gradient = vec![0.0; full.ncols()];
                full.apply_adjoint(&r, &mut gradient)?;
                let diff: Vec<f64> = r
                    .iter()
                    .zip(&baseline.demeaned)
                    .map(|(x, y)| x - y)
                    .collect();
                println!(
                    "{path},{name},{},{elapsed},{},{},{},{},{}",
                    setup + extra,
                    result.iterations,
                    result.converged,
                    norm(&gradient) / norm(&initial),
                    norm(&diff) / norm(&p.y),
                    if name.ends_with("coarse") {
                        coarse.active
                    } else {
                        0
                    }
                );
            }
            continue;
        }
        let mut initial_gradient = vec![0.0; a.ncols()];
        a.apply_adjoint(&rhs, &mut initial_gradient)?;
        let mut gradient = vec![0.0; a.ncols()];
        a.apply_adjoint(&baseline.demeaned, &mut gradient)?;
        println!(
            "{path},channel_pairs,{setup},{elapsed},{},{},{},0,0",
            baseline.iterations,
            baseline.converged,
            norm(&gradient) / norm(&initial_gradient)
        );
        for (name, m, extra) in [
            ("schur_diagonal", &diagonal as &dyn Operator, 0.0),
            ("schur_blocks", &blocks as &dyn Operator, 0.0),
            (
                "schur_blocks_coarse",
                &balanced as &dyn Operator,
                setup_coarse,
            ),
        ] {
            let start = Instant::now();
            let result = mlsmr(
                &a,
                &rhs,
                m,
                tol,
                20_000,
                MlsmrOptions {
                    local_size: Some(10),
                    reference_norm: Some(norm(&p.y)),
                    ..Default::default()
                },
            )?;
            let elapsed = start.elapsed().as_secs_f64();
            let mut r = vec![0.0; a.nrows()];
            a.apply(&result.x, &mut r)?;
            for (v, &b) in r.iter_mut().zip(&rhs) {
                *v = b - *v;
            }
            a.apply_adjoint(&r, &mut gradient)?;
            let diff: Vec<f64> = r
                .iter()
                .zip(&baseline.demeaned)
                .map(|(x, y)| x - y)
                .collect();
            println!(
                "{path},{name},{},{elapsed},{},{},{},{},{}",
                setup_blocks + extra,
                result.iterations,
                result.converged,
                norm(&gradient) / norm(&initial_gradient),
                norm(&diff) / norm(&p.y),
                if extra > 0.0 { coarse.active } else { 0 }
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_augmentation_matches_explicit_harmonic_lift() {
        let p = panel(1e-3);
        let a = Reduced::new(&p);
        let solver = Solver::new(
            vec![
                Effect::new(&p.worker, true, [&p.worker_slope[..]]).unwrap(),
                Effect::new(&p.firm, true, [&p.firm_slope[..]]).unwrap(),
            ],
            None,
            PreconditionerConfig::Additive {
                local_solver: Default::default(),
                reduction: Default::default(),
            },
        )
        .unwrap();
        let pairs = PairBasis::new(solver.preconditioner().unwrap(), &a);
        let blocks = Blocks::inverse(&a.blocks(), true);
        let coupling = Coupling::new(&a);
        let augmented = AugmentedPairs {
            base: &pairs,
            coupling: &coupling,
            correction: &blocks,
        };
        let nw = 2 * a.worker.n;
        let x: Vec<f64> = (0..pairs.nrows()).map(|i| (i as f64).sin()).collect();
        let mut observed = vec![0.0; x.len()];
        augmented.apply(&x, &mut observed).unwrap();
        let mut base = vec![0.0; x.len()];
        pairs.apply(&x, &mut base).unwrap();
        let mut rows = vec![0.0; a.nrows()];
        a.worker.gather(&x[..nw], &mut rows);
        let mut rhs = vec![0.0; a.ncols()];
        a.firm.scatter(&rows, &mut rhs);
        for (r, &v) in rhs.iter_mut().zip(&x[nw..]) {
            *r = v - *r;
        }
        let mut z = vec![0.0; a.ncols()];
        blocks.apply(&rhs, &mut z).unwrap();
        a.firm.gather(&z, &mut rows);
        let mut expected = vec![0.0; x.len()];
        a.worker.scatter(&rows, &mut expected[..nw]);
        for v in &mut expected[..nw] {
            *v = -*v;
        }
        expected[nw..].copy_from_slice(&z);
        let error: Vec<f64> = observed
            .iter()
            .zip(&base)
            .zip(&expected)
            .map(|((o, b), e)| o - b - e)
            .collect();
        assert!(norm(&error) < 1e-12 * norm(&expected));
        assert!(x.iter().zip(&expected).map(|(x, y)| x * y).sum::<f64>() >= 0.0);
    }

    fn panel(perturbation: f64) -> Panel {
        let mut p = Panel {
            worker: vec![],
            firm: vec![],
            worker_slope: vec![],
            firm_slope: vec![],
            y: vec![],
        };
        for w in 0..12 {
            for t in 0..10 {
                p.worker.push(w);
                p.firm.push((w + u32::from(t >= 7)) % 5);
                p.worker_slope.push(t as f64);
                p.firm_slope
                    .push(t as f64 + perturbation * ((13 * w + t) as f64).sin());
                p.y.push(((7 * w + 3 * t) as f64).cos());
            }
        }
        p
    }

    #[test]
    fn projected_operator_and_local_blocks_match_explicit_columns() {
        let a = Reduced::new(&panel(1e-3));
        let mut columns = Vec::new();
        for j in 0..a.ncols() {
            let mut e = vec![0.0; a.ncols()];
            e[j] = 1.0;
            let mut col = vec![0.0; a.nrows()];
            a.apply(&e, &mut col).unwrap();
            let mut twice = col.clone();
            a.project(&mut twice);
            assert!(
                norm(
                    &twice
                        .iter()
                        .zip(&col)
                        .map(|(x, y)| x - y)
                        .collect::<Vec<_>>()
                ) < 1e-13
            );
            columns.push(col);
        }
        let blocks = a.blocks();
        for (j, col) in columns.iter().enumerate() {
            let mut gram_column = vec![0.0; a.ncols()];
            a.apply_adjoint(col, &mut gram_column).unwrap();
            for (k, other) in columns.iter().enumerate() {
                let dot: f64 = col.iter().zip(other).map(|(x, y)| x * y).sum();
                assert!((gram_column[k] - dot).abs() < 1e-13);
                if j / 2 == k / 2 {
                    assert!((blocks[j / 2][j % 2 + k % 2] - dot).abs() < 1e-13);
                }
            }
        }
    }

    #[test]
    fn coarse_correction_is_symmetric_and_normalized() {
        let p = panel(1e-3);
        let a = Reduced::new(&p);
        let blocks = Blocks::inverse(&a.blocks(), true);
        let coarse = Coarse::new(&a, &p.worker_slope);
        assert_eq!(coarse.active, 1);
        assert!(
            (coarse
                .q
                .iter()
                .zip(&coarse.sq)
                .map(|(q, sq)| q * sq)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-7
        );
        let balanced = Balanced {
            blocks: &blocks,
            coarse: &coarse,
        };
        let mut matrix = vec![vec![0.0; a.ncols()]; a.ncols()];
        for (j, column) in matrix.iter_mut().enumerate() {
            let mut e = vec![0.0; a.ncols()];
            e[j] = 1.0;
            balanced.apply(&e, column).unwrap();
        }
        for j in 0..a.ncols() {
            assert!(matrix[j][j] > 0.0);
            for k in 0..a.ncols() {
                assert!(
                    (matrix[j][k] - matrix[k][j]).abs()
                        < 1e-10 * matrix[j][j].sqrt() * matrix[k][k].sqrt()
                );
            }
        }
        let exact = panel(0.0);
        let a = Reduced::new(&exact);
        assert_eq!(Coarse::new(&a, &exact.worker_slope).active, 0);
    }
}
