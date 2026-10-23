//! The reduced (Schur) solve prototype: the largest factor eliminated, mlsmr on the rest.
// `#[pyclass]` on a `Copy` enum expands to a `.clone()`.
#![allow(clippy::clone_on_copy)]

use std::time::Instant;

use pyo3::prelude::*;

use within::{
    BuildWarning, Design, LocalSolverConfig, LsmrOptions, PreconditionerConfig, ReductionStrategy,
    SolveResult, Solver, WithinError,
};

use crate::api::{extract_design, DesignSource};
use crate::config::{resolve_lsmr_config, PyLocalSolverConfig, PyReductionStrategy};
use crate::convert::{coerce_to_slice, readonly_f64_1d};
use crate::results::{run_solve_with_warnings, PySolveResult};

/// Which local inverses precondition the reduced system.
#[pyclass(eq, eq_int, frozen, from_py_object, module = "within._within")]
#[pyo3(name = "ReducedInverse")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyReducedInverse {
    /// Every factor pair: Schur complements of the eliminated factor's pairs plus the rest's Grams.
    PairSchur = 0,
    /// Only the eliminated factor's pairs, as block Jacobi on the reduced system.
    BlockJacobi = 1,
}

fn build_and_solve_reduced(
    design: Design<'_>,
    y: &[f64],
    weights: Option<&[f64]>,
    lsmr: &LsmrOptions,
    local_solver: &LocalSolverConfig,
    reduction: ReductionStrategy,
    inverse: PyReducedInverse,
) -> Result<(SolveResult, Vec<BuildWarning>), WithinError> {
    let t_start = Instant::now();
    let solver = Solver::new(design, weights, PreconditionerConfig::Off)?;
    let reduced = match inverse {
        PyReducedInverse::PairSchur => solver.pair_schur_prototype(local_solver, reduction),
        PyReducedInverse::BlockJacobi => solver.pair_schur_block_jacobi(local_solver, reduction),
    }?;
    let time_setup = t_start.elapsed().as_secs_f64();
    let mut result = reduced.solve_lsmr(y, lsmr)?;
    result.time_setup += time_setup;
    result.time_total = t_start.elapsed().as_secs_f64();
    let warnings = result.warnings.clone();
    Ok((result, warnings))
}

#[pyfunction]
#[pyo3(signature = (
    design, y, weights=None, options=None, local_solver=None,
    reduction=PyReductionStrategy::Auto, inverse=PyReducedInverse::PairSchur,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_reduced<'py>(
    py: Python<'py>,
    design: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
    options: Option<&Bound<'py, PyAny>>,
    local_solver: Option<PyLocalSolverConfig>,
    reduction: PyReductionStrategy,
    inverse: PyReducedInverse,
) -> PyResult<PySolveResult> {
    let params = resolve_lsmr_config(options)?;
    let local_solver = local_solver.unwrap_or_default().to_native();
    let reduction = reduction.to_native();
    let y = readonly_f64_1d("y", y)?;
    let weights = weights.map(|w| readonly_f64_1d("weights", w)).transpose()?;
    let y_arr = y.as_array();
    let w_view = weights.as_ref().map(|w| w.as_array());
    let solve = |design: Design<'_>| {
        let y_cow = coerce_to_slice(&y_arr);
        let w_cow = w_view.as_ref().map(coerce_to_slice);
        let (lsmr, local_solver) = (&params, &local_solver);
        build_and_solve_reduced(
            design,
            &y_cow,
            w_cow.as_deref(),
            lsmr,
            local_solver,
            reduction,
            inverse,
        )
    };
    match extract_design(py, design)? {
        DesignSource::Persistent(design) => run_solve_with_warnings(py, move || solve(design)),
        DesignSource::Categories(categories) => {
            let cats = categories.as_array();
            run_solve_with_warnings(py, move || solve(Design::from_categories(cats)?))
        }
        DesignSource::Effects(terms) => run_solve_with_warnings(py, move || {
            let effects: Vec<_> = terms.iter().map(|term| term.get().as_effect()).collect();
            solve(Design::new(effects)?)
        }),
    }
}
