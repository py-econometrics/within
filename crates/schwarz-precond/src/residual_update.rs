use crate::Operator;

/// Trait for updating the global residual after a single subdomain correction
/// in a multiplicative Schwarz sweep.
///
/// The multiplicative Schwarz preconditioner needs to update r_work after each
/// subdomain solve. Different strategies exist:
/// - `OperatorResidualUpdater`: naive full recompute via r = b - A*y_accum (O(n) per update)
/// - Observation-space updater (in `within` crate): exploits FE structure for sparse updates
pub trait ResidualUpdater: Send + Sync {
    /// Update the working residual after a subdomain correction.
    ///
    /// `global_indices`: the DOF indices touched by this subdomain
    /// `weighted_correction`: the PoU-weighted local correction (same length as `global_indices`)
    /// `r_work`: the full global residual vector to update in-place
    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]);

    /// Reset before a new sweep. Called at the start of each forward/backward sweep.
    fn reset(&mut self, r_original: &[f64]);
}

/// Naive residual updater that recomputes r = b - A * y_accum after each subdomain.
///
/// Maintains an internal accumulator y_accum. On each `update()`:
/// 1. Scatter `weighted_correction` into y_accum at `global_indices`
/// 2. Recompute `r_work = r_original - A * y_accum`
///
/// This is O(n_dofs) per update — correct but slow. Useful as a testing baseline.
pub struct OperatorResidualUpdater<'a, A: Operator> {
    operator: &'a A,
    /// Accumulator: sum of all corrections applied so far
    y_accum: Vec<f64>,
    /// Original residual (r before the sweep started)
    r_original: Vec<f64>,
    /// Scratch buffer for A * y_accum
    a_y: Vec<f64>,
}

impl<'a, A: Operator> OperatorResidualUpdater<'a, A> {
    /// Create a new updater.
    pub fn new(operator: &'a A, n_dofs: usize) -> Self {
        Self {
            operator,
            y_accum: vec![0.0; n_dofs],
            r_original: vec![0.0; n_dofs],
            a_y: vec![0.0; n_dofs],
        }
    }
}

impl<A: Operator> ResidualUpdater for OperatorResidualUpdater<'_, A> {
    fn reset(&mut self, r_original: &[f64]) {
        self.r_original.copy_from_slice(r_original);
        self.y_accum.iter_mut().for_each(|v| *v = 0.0);
    }

    fn update(&mut self, global_indices: &[u32], weighted_correction: &[f64], r_work: &mut [f64]) {
        // 1. Scatter correction into accumulator
        for (k, &gi) in global_indices.iter().enumerate() {
            self.y_accum[gi as usize] += weighted_correction[k];
        }

        // 2. Recompute r_work = r_original - A * y_accum
        self.operator.apply(&self.y_accum, &mut self.a_y);
        for (r, (&ro, &ay)) in r_work
            .iter_mut()
            .zip(self.r_original.iter().zip(self.a_y.iter()))
        {
            *r = ro - ay;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple diagonal operator for testing: A = diag(values)
    struct DiagOperator {
        values: Vec<f64>,
    }

    impl DiagOperator {
        fn new(values: Vec<f64>) -> Self {
            Self { values }
        }
    }

    impl Operator for DiagOperator {
        fn nrows(&self) -> usize {
            self.values.len()
        }
        fn ncols(&self) -> usize {
            self.values.len()
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            for i in 0..self.values.len() {
                y[i] = self.values[i] * x[i];
            }
        }
        fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
            self.apply(x, y);
        }
    }

    #[test]
    fn test_operator_residual_updater_basic() {
        // A = diag(2, 3, 1, 4)
        let a = DiagOperator::new(vec![2.0, 3.0, 1.0, 4.0]);
        let mut updater = OperatorResidualUpdater::new(&a, 4);

        // r_original = [10, 12, 5, 8]
        let r_original = [10.0, 12.0, 5.0, 8.0];
        updater.reset(&r_original);

        let mut r_work = r_original.to_vec();

        // First correction: indices [0, 2], correction [1.0, 2.0]
        // y_accum becomes [1, 0, 2, 0]
        // A * y_accum = [2, 0, 2, 0]
        // r_work = [10-2, 12-0, 5-2, 8-0] = [8, 12, 3, 8]
        updater.update(&[0, 2], &[1.0, 2.0], &mut r_work);
        assert!((r_work[0] - 8.0).abs() < 1e-12);
        assert!((r_work[1] - 12.0).abs() < 1e-12);
        assert!((r_work[2] - 3.0).abs() < 1e-12);
        assert!((r_work[3] - 8.0).abs() < 1e-12);

        // Second correction: indices [1, 3], correction [0.5, 1.0]
        // y_accum becomes [1, 0.5, 2, 1]
        // A * y_accum = [2, 1.5, 2, 4]
        // r_work = [10-2, 12-1.5, 5-2, 8-4] = [8, 10.5, 3, 4]
        updater.update(&[1, 3], &[0.5, 1.0], &mut r_work);
        assert!((r_work[0] - 8.0).abs() < 1e-12);
        assert!((r_work[1] - 10.5).abs() < 1e-12);
        assert!((r_work[2] - 3.0).abs() < 1e-12);
        assert!((r_work[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_operator_residual_updater_reset() {
        let a = DiagOperator::new(vec![1.0, 1.0]);
        let mut updater = OperatorResidualUpdater::new(&a, 2);

        // First sweep
        updater.reset(&[5.0, 3.0]);
        let mut r = vec![5.0, 3.0];
        updater.update(&[0], &[2.0], &mut r);
        assert!((r[0] - 3.0).abs() < 1e-12);

        // Reset for second sweep — accumulator should be cleared
        updater.reset(&[10.0, 7.0]);
        let mut r = vec![10.0, 7.0];
        updater.update(&[1], &[1.0], &mut r);
        // y_accum = [0, 1], A*y = [0, 1], r = [10-0, 7-1] = [10, 6]
        assert!((r[0] - 10.0).abs() < 1e-12);
        assert!((r[1] - 6.0).abs() < 1e-12);
    }
}
