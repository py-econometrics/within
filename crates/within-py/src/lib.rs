mod api;

use pyo3::prelude::*;

#[pymodule]
fn _within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<api::PySolveResult>()?;
    m.add_class::<api::PyLSMR>()?;
    m.add_class::<api::PyCG>()?;
    m.add_class::<api::PyOneLevelSchwarz>()?;
    m.add_class::<api::PyMultiplicativeOneLevelSchwarz>()?;
    m.add_class::<api::PyGMRES>()?;

    m.add_class::<api::PyApproxCholConfig>()?;
    m.add_class::<api::PyApproxSchurConfig>()?;
    m.add_function(wrap_pyfunction!(api::py_solve, m)?)?;
    Ok(())
}
