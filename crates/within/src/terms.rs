//! Fixed-effects term inputs: the [`Fe`] factor specification.

/// One fixed-effects factor term: a per-observation level index, optional
/// varying-slope loadings, and whether to include the level intercept.
// Bridge: fields are read once `Fe` lowers to a store; remove when wired.
#[allow(dead_code)]
pub struct Fe {
    levels: Vec<u32>,
    slopes: Vec<Vec<f64>>,
    intercept: bool,
}
