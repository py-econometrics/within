use super::{CoefficientLayout, Elimination, LsmrOptions, Run, SolveResult, Solver};
use crate::linalg::dot;
use crate::{SolveError, WithinError};
use rustc_hash::FxHashMap;
use schwarz_precond::{LsmrResult, LsmrStopReason, Operator};
use std::time::Instant;

fn invalid(message: impl Into<String>) -> SolveError {
    SolveError::InvalidInput {
        context: "Schur PCG",
        message: message.into(),
    }
}

fn to_u32(x: usize) -> u32 {
    u32::try_from(x).expect("CSR index exceeds u32::MAX")
}

/// Rectangular CSR, column indices ascending within each row.
pub(super) struct CsrBlock {
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) data: Vec<f64>,
}

impl CsrBlock {
    pub(super) fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (start, end) = (self.indptr[i] as usize, self.indptr[i + 1] as usize);
        (self.indices[start..end].iter())
            .zip(&self.data[start..end])
            .map(|(&j, &w)| (j as usize, w))
    }
}

pub(super) struct Assembly {
    pub(super) rows: Vec<FxHashMap<usize, f64>>,
}

impl Assembly {
    pub(super) fn new(nrows: usize) -> Self {
        Self {
            rows: (0..nrows).map(|_| FxHashMap::default()).collect(),
        }
    }

    pub(super) fn add(&mut self, i: usize, j: usize, value: f64) {
        *self.rows[i].entry(j).or_default() += value;
    }

    pub(super) fn finish(self) -> CsrBlock {
        let nrows = self.rows.len();
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for row in self.rows {
            let mut entries: Vec<_> = row.into_iter().filter(|(_, v)| *v != 0.0).collect();
            entries.sort_unstable_by_key(|&(j, _)| j);
            for (j, v) in entries {
                indices.push(to_u32(j));
                data.push(v);
            }
            indptr.push(to_u32(indices.len()));
        }
        CsrBlock {
            indptr,
            indices,
            data,
        }
    }
}

pub(super) struct SchurMatrix {
    pub(super) kept: CsrBlock,
    pub(super) coupling: CsrBlock,
    pub(super) inverse: Vec<f64>,
}

impl SchurMatrix {
    pub(super) fn apply(&self, x: &[f64], y: &mut [f64], eliminated: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = self.kept.row(i).map(|(j, a)| a * x[j]).sum();
        }
        for (i, ti) in eliminated.iter_mut().enumerate() {
            *ti = self.inverse[i] * self.coupling.row(i).map(|(j, a)| a * x[j]).sum::<f64>();
            for (j, a) in self.coupling.row(i) {
                y[j] -= a * *ti;
            }
        }
    }
}

impl SchurMatrix {
    #[cfg(feature = "pair-schur-prototype")]
    pub(super) fn build(e: &Elimination<'_>) -> Result<Self, SolveError> {
        e.require_intercepts("pair_schur_prototype")?;
        let mut h = Assembly::new(e.n_kept());
        let mut c = Assembly::new(e.removed.len());
        let mut columns = Vec::with_capacity(e.prepared.design.terms.len() - 1);
        for (i, &level) in e.prepared.design.terms[e.term].levels().iter().enumerate() {
            let w = e.scale(i).powi(2);
            if w == 0.0 {
                continue;
            }
            columns.clear();
            let mut offset = 0;
            for (index, term) in e.prepared.design.terms.iter().enumerate() {
                if index == e.term {
                    continue;
                }
                let column = offset + term.levels()[i] as usize;
                c.add(level as usize, column, w);
                columns.push(column);
                offset += term.n_dofs();
            }
            for &a in &columns {
                for &b in &columns {
                    h.add(a, b, w);
                }
            }
        }
        Ok(Self {
            kept: h.finish(),
            coupling: c.finish(),
            inverse: e.inverse.clone(),
        })
    }
}

pub(super) fn solve_reduced(
    solver: &Solver<'_>,
    e: &Elimination<'_>,
    matrix: &SchurMatrix,
    y: &[f64],
    options: &LsmrOptions,
    inverse: &dyn Operator,
) -> Result<SolveResult, WithinError> {
    if !options.tol.is_finite() || options.tol <= 0.0 {
        return Err(invalid("tol must be finite and positive").into());
    }
    let start = Instant::now();
    let rhs = solver.prepare(y)?;
    let mut full_rhs = vec![0.0; solver.n_dofs()];
    rhs.op.apply_adjoint(rhs.b(), &mut full_rhs)?;
    let reference = dot(&full_rhs, &full_rhs).sqrt();
    let mut g = vec![0.0; e.n_kept()];
    e.restrict(&full_rhs, &mut g);
    for (i, &inverse) in e.inverse.iter().enumerate() {
        let eliminated = inverse * full_rhs[e.removed.start + i];
        for (j, a) in matrix.coupling.row(i) {
            g[j] -= a * eliminated;
        }
    }
    let setup_seconds = start.elapsed().as_secs_f64();
    let solve_start = Instant::now();
    let mut x = vec![0.0; g.len()];
    let mut r = g.clone();
    let mut z = vec![0.0; g.len()];
    let mut p = vec![0.0; g.len()];
    let mut ap = vec![0.0; g.len()];
    let mut scratch = vec![0.0; e.removed.len()];
    let threshold = 0.5 * options.tol * reference;
    let mut iterations = 0;
    let mut rz_previous = 0.0;
    let mut restart = true;
    while dot(&r, &r).sqrt() > threshold && iterations < options.maxiter {
        inverse.apply(&r, &mut z)?;
        let rz = dot(&r, &z);
        if !rz.is_finite() || rz <= 0.0 {
            break;
        }
        if restart {
            p.copy_from_slice(&z);
        } else {
            let beta = rz / rz_previous;
            for (pi, &zi) in p.iter_mut().zip(&z) {
                *pi = zi + beta * *pi;
            }
        }
        matrix.apply(&p, &mut ap, &mut scratch);
        let curvature = dot(&p, &ap);
        if !curvature.is_finite() || curvature <= 0.0 {
            break;
        }
        let alpha = rz / curvature;
        for (((xi, ri), &pi), &api) in x.iter_mut().zip(&mut r).zip(&p).zip(&ap) {
            *xi += alpha * pi;
            *ri -= alpha * api;
        }
        iterations += 1;
        restart = iterations % 64 == 0 || dot(&r, &r).sqrt() <= threshold;
        if restart {
            matrix.apply(&x, &mut ap, &mut scratch);
            for ((ri, &gi), &axi) in r.iter_mut().zip(&g).zip(&ap) {
                *ri = gi - axi;
            }
        }
        rz_previous = rz;
    }
    let mut full = vec![0.0; solver.n_dofs()];
    e.embed(&x, &mut full);
    for (i, &inverse) in e.inverse.iter().enumerate() {
        full[e.removed.start + i] = inverse
            * (full_rhs[e.removed.start + i]
                - matrix.coupling.row(i).map(|(j, a)| a * x[j]).sum::<f64>());
    }
    let mut residual = vec![0.0; rhs.b().len()];
    rhs.op.apply(&full, &mut residual)?;
    for (ri, &bi) in residual.iter_mut().zip(rhs.b()) {
        *ri = bi - *ri;
    }
    let residual_norm = dot(&residual, &residual).sqrt();
    rhs.op.apply_adjoint(&residual, &mut full_rhs)?;
    let gradient_norm = dot(&full_rhs, &full_rhs).sqrt();
    let converged = gradient_norm <= options.tol * reference;
    let ratio = if reference > 0.0 {
        gradient_norm / reference
    } else {
        gradient_norm
    };
    let result = LsmrResult {
        x: full,
        converged,
        iterations,
        residual_norm,
        normal_eq_residual: ratio,
        true_residual: Some(residual),
        stop_reason: if converged {
            LsmrStopReason::NormalEquationTolerance
        } else {
            LsmrStopReason::MaxIterations
        },
    };
    let solution = solver.finish(
        rhs,
        Run {
            result,
            solve_secs: solve_start.elapsed().as_secs_f64(),
        },
    );
    Ok(SolveResult {
        x: solution.x,
        unidentified: solver.unidentified(),
        warnings: Vec::new(),
        layout: CoefficientLayout::from_design(&solver.prepared.design),
        demeaned: solution.demeaned,
        converged: solution.converged,
        iterations: solution.iterations,
        residual: solution.residual,
        time_total: start.elapsed().as_secs_f64(),
        time_setup: setup_seconds,
        time_solve: solution.time_solve,
    })
}

pub(super) fn solve_with_full_preconditioner(
    solver: &Solver<'_>,
    e: &Elimination<'_>,
    matrix: &SchurMatrix,
    y: &[f64],
    options: &LsmrOptions,
    full: &crate::Preconditioner,
) -> Result<SolveResult, WithinError> {
    if full.nrows() != solver.n_dofs() || full.ncols() != solver.n_dofs() {
        return Err(invalid("full preconditioner dimensions do not match this design").into());
    }
    let start = Instant::now();
    let inverse = super::RestrictedInverse {
        elimination: e,
        full,
        scratch: std::sync::Mutex::new((vec![0.0; solver.n_dofs()], vec![0.0; solver.n_dofs()])),
    };
    let mut result = solve_reduced(solver, e, matrix, y, options, &inverse)?;
    let elapsed = start.elapsed().as_secs_f64();
    result.time_setup += elapsed - result.time_total;
    result.time_total = elapsed;
    Ok(result)
}
