mod api;

use pyo3::prelude::*;

#[pymodule]
fn _within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<api::PySolveResult>()?;
    m.add_class::<api::PyCG>()?;
    m.add_class::<api::PyGMRES>()?;
    m.add_class::<api::PyAdditiveSchwarz>()?;
    m.add_class::<api::PyMultiplicativeSchwarz>()?;
    m.add_class::<api::PyOperatorRepr>()?;
    m.add_class::<api::PyApproxCholConfig>()?;
    m.add_class::<api::PyApproxSchurConfig>()?;
    m.add_class::<api::PySchurComplement>()?;
    m.add_class::<api::PyFullSddm>()?;
    m.add_function(wrap_pyfunction!(api::solve, m)?)?;
    Ok(())
}
