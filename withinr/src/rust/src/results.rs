//! Native result → R list conversions.

use extendr_api::prelude::*;

use within::{BatchSolveResult, SolveResult};

use crate::convert::usize_to_i32;

pub(crate) fn result_to_list(result: SolveResult) -> Result<List> {
    Ok(list!(
        x = result.x,
        demeaned = result.demeaned,
        converged = result.converged,
        iterations = usize_to_i32(result.iterations, "iterations")?,
        residual = result.residual,
        time_total = result.time_total,
        time_setup = result.time_setup,
        time_solve = result.time_solve
    ))
}

pub(crate) fn batch_result_to_list(result: BatchSolveResult) -> Result<List> {
    let n_rhs = result.converged.len();

    let mut x = RMatrix::new(result.n_dofs, n_rhs);
    x.data_mut().copy_from_slice(&result.x);

    let mut demeaned = RMatrix::new(result.n_obs, n_rhs);
    demeaned.data_mut().copy_from_slice(&result.demeaned);

    let iterations = result
        .iterations
        .iter()
        .map(|&value| usize_to_i32(value, "iterations"))
        .collect::<Result<Vec<_>>>()?;

    Ok(list!(
        x = x,
        demeaned = demeaned,
        converged = result.converged,
        iterations = iterations,
        residual = result.residual,
        time_solve = result.time_solve,
        time_total = result.time_total
    ))
}
