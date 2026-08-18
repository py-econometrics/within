use super::NotScalable;
use crate::config::ScalingConfig;
use crate::csr_block::CsrBlock;
use crate::domain::CrossTab;
use crate::linalg::dot;

pub(super) struct DominanceScaling {
    pub(super) scales: Vec<f64>,
    pub(super) sweeps: usize,
    /// Largest relative dominance violation at hand-over, clamped at 0.
    pub(super) violation: f64,
}

#[derive(Clone, Copy)]
enum BipartiteSide {
    Rows,
    Columns,
}

struct NormalizedCrossOperator<'a> {
    small_to_large: &'a CsrBlock,
    large_to_small: &'a CsrBlock,
    small_inv_sqrt: &'a [f64],
    large_inv_sqrt: &'a [f64],
    small_side: BipartiteSide,
}

#[derive(Clone, Copy)]
enum CandidateKind {
    Strict,
    Boundary,
}

impl<'a> NormalizedCrossOperator<'a> {
    fn new(cross_tab: &'a CrossTab, inv_sqrt: &'a [f64]) -> Self {
        let (row_inv_sqrt, col_inv_sqrt) = inv_sqrt.split_at(cross_tab.n_rows());
        let (small_to_large, large_to_small, small_inv_sqrt, large_inv_sqrt, small_side) =
            if cross_tab.n_rows() <= cross_tab.n_cols() {
                (
                    &cross_tab.ct,
                    &cross_tab.c,
                    row_inv_sqrt,
                    col_inv_sqrt,
                    BipartiteSide::Rows,
                )
            } else {
                (
                    &cross_tab.c,
                    &cross_tab.ct,
                    col_inv_sqrt,
                    row_inv_sqrt,
                    BipartiteSide::Columns,
                )
            };
        Self {
            small_to_large,
            large_to_small,
            small_inv_sqrt,
            large_inv_sqrt,
            small_side,
        }
    }

    fn small_len(&self) -> usize {
        self.small_inv_sqrt.len()
    }

    fn large_len(&self) -> usize {
        self.large_inv_sqrt.len()
    }

    fn apply_large(&self, input: &[f64], output: &mut [f64]) {
        multiply_normalized(
            self.small_to_large,
            self.large_inv_sqrt,
            self.small_inv_sqrt,
            input,
            output,
        );
    }

    fn apply_small(&self, input: &[f64], output: &mut [f64]) {
        multiply_normalized(
            self.large_to_small,
            self.small_inv_sqrt,
            self.large_inv_sqrt,
            input,
            output,
        );
    }

    fn apply_reduced(&self, input: &[f64], large_work: &mut [f64], output: &mut [f64]) {
        // Eliminating the large side gives `I - MᵀM` on the cheaper side.
        self.apply_large(input, large_work);
        self.apply_small(large_work, output);
        for (value, &input_value) in output.iter_mut().zip(input) {
            *value = input_value - *value;
        }
    }

    fn reduced_rhs(&self, large_work: &mut [f64]) -> Vec<f64> {
        large_work.fill(1.0);
        let mut rhs = vec![0.0; self.small_len()];
        self.apply_small(large_work, &mut rhs);
        for value in &mut rhs {
            *value += 1.0;
        }
        rhs
    }

    fn candidate_violation(
        &self,
        small: &[f64],
        kind: CandidateKind,
        large: &mut [f64],
        small_image: &mut [f64],
    ) -> Option<f64> {
        if !small.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        self.apply_large(small, large);
        if matches!(kind, CandidateKind::Strict) {
            for value in large.iter_mut() {
                *value += 1.0;
            }
        }
        if !large.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        self.apply_small(large, small_image);
        Some(
            small_image
                .iter()
                .zip(small)
                .map(|(&target, &value)| target / value - 1.0)
                .fold(0.0f64, f64::max),
        )
    }

    fn materialize_candidate(
        &self,
        small: &[f64],
        large: &[f64],
        violation: f64,
    ) -> Result<ScalingCandidate, NotScalable> {
        let mut mu: Vec<f64> = match self.small_side {
            BipartiteSide::Rows => small.iter().chain(large).copied().collect(),
            BipartiteSide::Columns => large.iter().chain(small).copied().collect(),
        };
        let peak = mu.iter().copied().fold(0.0f64, f64::max);
        if !peak.is_finite() || peak <= 0.0 {
            return Err(NotScalable);
        }
        for value in mu.iter_mut() {
            *value /= peak;
        }
        if !mu.iter().all(|value| value.is_finite() && *value > 0.0) {
            return Err(NotScalable);
        }
        Ok(ScalingCandidate { mu, violation })
    }
}

fn multiply_normalized(
    matrix: &CsrBlock,
    output_inv_sqrt: &[f64],
    input_inv_sqrt: &[f64],
    input: &[f64],
    output: &mut [f64],
) {
    debug_assert_eq!(matrix.nrows, output.len());
    debug_assert_eq!(matrix.ncols, input.len());
    for (i, value) in output.iter_mut().enumerate() {
        *value = matrix
            .row(i)
            .map(|(j, entry)| entry.abs() * output_inv_sqrt[i] * input_inv_sqrt[j] * input[j])
            .sum();
    }
}

struct ScalingCandidate {
    mu: Vec<f64>,
    violation: f64,
}

struct ScalingResult {
    candidate: ScalingCandidate,
    iterations: usize,
}

#[derive(Clone, Copy)]
enum ReducedCgStep {
    Advanced,
    ResidualExhausted,
    NonPositiveCurvature,
}

struct ReducedCg {
    solution: Vec<f64>,
    residual: Vec<f64>,
    direction: Vec<f64>,
    reduced_image: Vec<f64>,
    residual_norm_squared: f64,
}

impl ReducedCg {
    fn new(rhs: Vec<f64>) -> Result<Self, NotScalable> {
        let dimension = rhs.len();
        let residual_norm_squared = dot(&rhs, &rhs);
        if !residual_norm_squared.is_finite() {
            return Err(NotScalable);
        }
        Ok(Self {
            solution: vec![0.0; dimension],
            residual: rhs.clone(),
            direction: rhs,
            reduced_image: vec![0.0; dimension],
            residual_norm_squared,
        })
    }

    fn step(
        &mut self,
        operator: &NormalizedCrossOperator<'_>,
        large_work: &mut [f64],
    ) -> Result<ReducedCgStep, NotScalable> {
        operator.apply_reduced(&self.direction, large_work, &mut self.reduced_image);
        let direction_norm_squared = dot(&self.direction, &self.direction);
        let curvature = dot(&self.direction, &self.reduced_image);
        if !direction_norm_squared.is_finite() || !curvature.is_finite() {
            return Err(NotScalable);
        }
        let slack = 64.0 * f64::EPSILON * direction_norm_squared;
        if curvature <= slack {
            return Ok(ReducedCgStep::NonPositiveCurvature);
        }

        let alpha = self.residual_norm_squared / curvature;
        if !alpha.is_finite() {
            return Err(NotScalable);
        }
        for ((value, residual), (&direction, &image)) in self
            .solution
            .iter_mut()
            .zip(&mut self.residual)
            .zip(self.direction.iter().zip(&self.reduced_image))
        {
            *value += alpha * direction;
            *residual -= alpha * image;
        }

        let next_residual_norm_squared = dot(&self.residual, &self.residual);
        if !next_residual_norm_squared.is_finite() {
            return Err(NotScalable);
        }
        if next_residual_norm_squared == 0.0 {
            return Ok(ReducedCgStep::ResidualExhausted);
        }
        let beta = next_residual_norm_squared / self.residual_norm_squared;
        for (direction, &residual) in self.direction.iter_mut().zip(&self.residual) {
            *direction = residual + beta * *direction;
        }
        self.residual_norm_squared = next_residual_norm_squared;
        Ok(ReducedCgStep::Advanced)
    }
}

fn reduced_cg_scaling(
    operator: &NormalizedCrossOperator<'_>,
    initial: ScalingCandidate,
    scaling: &ScalingConfig,
) -> Result<ScalingResult, NotScalable> {
    let mut large_work = vec![0.0; operator.large_len()];
    let rhs = operator.reduced_rhs(&mut large_work);
    let mut cg = ReducedCg::new(rhs)?;
    let mut best = initial;
    let mut small_image = vec![0.0; operator.small_len()];

    for iteration in 1..=scaling.max_sweeps {
        let step = cg.step(operator, &mut large_work)?;
        if matches!(step, ReducedCgStep::NonPositiveCurvature) {
            // A singular boundary exposes its Perron vector as the zero-curvature direction.
            let boundary_small: Vec<f64> = cg.direction.iter().map(|value| value.abs()).collect();
            if let Some(violation) = operator.candidate_violation(
                &boundary_small,
                CandidateKind::Boundary,
                &mut large_work,
                &mut small_image,
            ) {
                if violation < best.violation {
                    if let Ok(candidate) =
                        operator.materialize_candidate(&boundary_small, &large_work, violation)
                    {
                        best = candidate;
                    }
                }
            }
            return Ok(ScalingResult {
                candidate: best,
                iterations: iteration,
            });
        }

        // A certificate check costs two extra passes; past the cheap early exits, probe on a stride.
        let last =
            matches!(step, ReducedCgStep::ResidualExhausted) || iteration == scaling.max_sweeps;
        if last || iteration <= 8 || iteration % 4 == 0 {
            if let Some(violation) = operator.candidate_violation(
                &cg.solution,
                CandidateKind::Strict,
                &mut large_work,
                &mut small_image,
            ) {
                if violation < best.violation {
                    if let Ok(candidate) =
                        operator.materialize_candidate(&cg.solution, &large_work, violation)
                    {
                        best = candidate;
                    }
                }
                if best.violation <= scaling.tolerance {
                    return Ok(ScalingResult {
                        candidate: best,
                        iterations: iteration,
                    });
                }
            }
        }

        if matches!(step, ReducedCgStep::ResidualExhausted) {
            return Ok(ScalingResult {
                candidate: best,
                iterations: iteration,
            });
        }
    }
    Ok(ScalingResult {
        candidate: best,
        iterations: scaling.max_sweeps,
    })
}

pub(super) fn dominance_scaling(
    cross_tab: &CrossTab,
    diagonal: &[f64],
    scaling: &ScalingConfig,
) -> Result<DominanceScaling, NotScalable> {
    let n = cross_tab.n_local();
    debug_assert_eq!(diagonal.len(), n);
    if diagonal.iter().any(|d| !d.is_finite() || *d <= 0.0) {
        return Err(NotScalable);
    }

    let already_dominant = (0..n).all(|i| {
        let row_sum: f64 = cross_tab.neighbors(i).map(|(_, value)| value.abs()).sum();
        row_sum <= diagonal[i] * (1.0 + scaling.tolerance)
    });
    if already_dominant {
        return Ok(DominanceScaling {
            scales: vec![1.0; n],
            sweeps: 0,
            violation: 0.0,
        });
    }

    let inv_sqrt: Vec<f64> = diagonal.iter().map(|d| 1.0 / d.sqrt()).collect();
    let violation_of = |mu: &[f64]| {
        (0..n)
            .map(|i| {
                let target: f64 = cross_tab
                    .neighbors(i)
                    .map(|(j, value)| value.abs() * inv_sqrt[i] * inv_sqrt[j] * mu[j])
                    .sum();
                target / mu[i] - 1.0
            })
            .fold(0.0f64, f64::max)
    };

    let initial_mu = vec![1.0; n];
    let initial = ScalingCandidate {
        violation: violation_of(&initial_mu),
        mu: initial_mu,
    };
    let operator = NormalizedCrossOperator::new(cross_tab, &inv_sqrt);
    let result = reduced_cg_scaling(&operator, initial, scaling)?;
    let scales: Vec<f64> = result
        .candidate
        .mu
        .iter()
        .zip(inv_sqrt.iter())
        .map(|(&value, &normalizer)| value * normalizer)
        .collect();
    if !scales.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err(NotScalable);
    }
    Ok(DominanceScaling {
        scales,
        sweeps: result.iterations,
        violation: result.candidate.violation.max(0.0),
    })
}
