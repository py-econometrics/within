//! A slope pair's [`Grounding`], read from its observations rather than its rounded Gram.

use crate::csr_block::CsrBlock;
use crate::domain::cross_tab::{BipartiteComponent, Loading, PairColumns};

use super::Grounding;

// `ρ ≥ λ_min`, so a well-conditioned block grounds; an exact alias reads `ρ ~ ε²`.
const FLOATING_RAYLEIGH_BOUND: f64 = 1e-13;

/// Union-find over the levels `[rows | cols]` of `C`, scaled so `|x|·u_i = |y|·u_j` on each join.
pub(super) struct ScaledForest {
    n_rows: usize,
    parent: Vec<u32>,
    /// `u_node / u_parent`: the null vector along the joins, when one exists.
    scale: Vec<f64>,
    size: Vec<u32>,
    /// Per root: its set's `uᵀ(D − |C|)u` and `uᵀDu`, in the root's units.
    numerator: Vec<f64>,
    denominator: Vec<f64>,
}

impl ScaledForest {
    /// One pass over the observations, then one over the cells for their cancellation.
    pub(super) fn grow<Lq: Loading, Lr: Loading>(
        columns: &PairColumns<'_, Lq, Lr>,
        c: &CsrBlock,
    ) -> Self {
        let n_local = c.nrows + c.ncols;
        let mut forest = Self {
            n_rows: c.nrows,
            parent: (0..n_local as u32).collect(),
            scale: vec![1.0; n_local],
            size: vec![1; n_local],
            numerator: vec![0.0; n_local],
            denominator: vec![0.0; n_local],
        };
        // Without a stored cell every level is alone, and its `0/0` quotient grounds it.
        if c.nnz() == 0 {
            return forest;
        }
        let mut magnitude = vec![0.0; c.nnz()];
        // uᵀ(D − |C|)u = Σ_o w(|x|u_i − |y|u_j)² + 2Σ_cells u_i·u_j(Σ|wxy| − |Σwxy|).
        // Weightless rows carry nothing, so they are skipped.
        let observations = (0..columns.n_obs()).map(|uid| columns.observation(uid));
        for o in observations.filter(|o| o.w != 0.0) {
            let (i, j) = (o.row, c.nrows + o.col);
            let cell = (o.x != 0.0 && o.y != 0.0)
                .then(|| c.position(o.row, o.col))
                .flatten();
            let ((ri, ui), (rj, uj)) = match cell {
                Some(p) => {
                    magnitude[p] += (o.w * o.x * o.y).abs();
                    forest.join(i, j, o.x.abs() / o.y.abs())
                }
                None => (forest.find(i), forest.find(j)),
            };
            let (a, b) = (o.x.abs() * ui, o.y.abs() * uj);
            if cell.is_some() {
                forest.numerator[ri] += o.w * (a - b) * (a - b);
            } else {
                forest.numerator[ri] += o.w * a * a;
                forest.numerator[rj] += o.w * b * b;
            }
            forest.denominator[ri] += o.w * a * a;
            forest.denominator[rj] += o.w * b * b;
        }
        for i in 0..c.nrows {
            let cells = c.indptr[i] as usize..c.indptr[i + 1] as usize;
            let row = c.indices[cells.clone()].iter().zip(&c.data[cells.clone()]);
            for ((&j, &value), &magnitude) in row.zip(&magnitude[cells]) {
                let cancelled = magnitude - value.abs();
                if cancelled > 0.0 {
                    let (root, ui) = forest.find(i);
                    let (_, uj) = forest.find(c.nrows + j as usize);
                    forest.numerator[root] += 2.0 * ui * uj * cancelled;
                }
            }
        }
        forest
    }

    /// The root of `node`'s set and `u_node` in its units, compressing the path.
    fn find(&mut self, node: usize) -> (usize, f64) {
        let parent = self.parent[node] as usize;
        if parent == node {
            return (node, 1.0);
        }
        // Union by size leaves most levels one link from their root; those need no rewrite.
        if self.parent[parent] as usize == parent {
            return (parent, self.scale[node]);
        }
        let (root, u_parent) = self.find(parent);
        let u = self.scale[node] * u_parent;
        (self.parent[node], self.scale[node]) = (root as u32, u);
        (root, u)
    }

    /// Joins the sets of `i` and `j` so that `u_j = ratio·u_i`, unless they already share one.
    fn join(&mut self, i: usize, j: usize, ratio: f64) -> ((usize, f64), (usize, f64)) {
        let ((ri, ui), (rj, uj)) = (self.find(i), self.find(j));
        if ri == rj {
            return ((ri, ui), (rj, uj));
        }
        // Union by size keeps the paths short; the absorbed root's sums change units by `s²`.
        let (root, absorbed, s, joined) = if self.size[ri] >= self.size[rj] {
            (ri, rj, ui * ratio / uj, ((ri, ui), (ri, ui * ratio)))
        } else {
            (rj, ri, uj / (ui * ratio), ((rj, uj / ratio), (rj, uj)))
        };
        (self.parent[absorbed], self.scale[absorbed]) = (root as u32, s);
        self.size[root] += self.size[absorbed];
        self.numerator[root] += self.numerator[absorbed] * s * s;
        self.denominator[root] += self.denominator[absorbed] * s * s;
        joined
    }

    /// Each set with its grounding, ordered by lowest level as the cross-tab's DFS orders them.
    pub(super) fn into_components(mut self) -> Vec<(BipartiteComponent, Grounding)> {
        let mut slot = vec![usize::MAX; self.parent.len()];
        let mut components: Vec<(BipartiteComponent, Grounding)> = Vec::new();
        for node in 0..self.parent.len() {
            let (root, _) = self.find(node);
            if slot[root] == usize::MAX {
                slot[root] = components.len();
                let empty = BipartiteComponent {
                    rows: vec![],
                    cols: vec![],
                };
                components.push((empty, self.grounding(root)));
            }
            let (component, _) = &mut components[slot[root]];
            if node < self.n_rows {
                component.rows.push(node);
            } else {
                component.cols.push(node - self.n_rows);
            }
        }
        components
    }

    /// `Floating` iff `ρ = uᵀ(D − |C|)u / uᵀDu ≤ λ*` over the set of `root`.
    fn grounding(&self, root: usize) -> Grounding {
        // An overflowed `u` reads NaN, so it grounds.
        if self.numerator[root] / self.denominator[root] <= FLOATING_RAYLEIGH_BOUND {
            Grounding::Floating
        } else {
            Grounding::Grounded
        }
    }
}

#[cfg(test)]
mod tests;
