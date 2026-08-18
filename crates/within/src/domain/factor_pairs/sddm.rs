//! Validated conversion from a signed bipartite Gram block to SDDM form.
//!
//! One signed diagonal coordinate map `S` makes `S A S` a Z-matrix and scales
//! it to diagonal dominance. Surplus is retained as explicit ground-edge
//! weight, so downstream reduction never has to infer rank or grounding from
//! rounded diagonals. Plain components take a fast path: their Laplacian claim
//! is validated and adopted without sign folding, scaling, or a data rewrite.
//! Frustrated components — where no signature exists — keep their single
//! *signed* matrix (dominance-scaled but not folded to a Z-matrix) and carry
//! a [`MatrixForm::SignedPendingCover`] marker; the Gremban cover that makes the
//! reduced Schur SDDM is built transiently at factor time (see
//! [`crate::block_elim`]), so the stored matrix stays single-sized.

use super::ComponentClass;
use crate::config::{ScalingConfig, ScalingFailure};
use crate::csr_block::CsrBlock;
use crate::domain::CrossTab;

#[derive(Clone, Copy)]
struct RoundoffBudget {
    ulps: f64,
}

impl RoundoffBudget {
    fn tolerance(self, n: usize, total_diagonal: f64) -> f64 {
        self.ulps * f64::EPSILON * (n.max(1) as f64).sqrt() * total_diagonal
    }
}

const LAPLACIAN_VALIDATION_BUDGET: RoundoffBudget = RoundoffBudget { ulps: 64.0 };
// A false Grounded retains noise edges; a false Floating deletes an identified direction.
const FLOATING_CLASSIFICATION_BUDGET: RoundoffBudget = RoundoffBudget { ulps: 4.0 };

/// Gauge of a reduced system: `Floating` anchors one node, `Grounded` factors the full complement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Grounding {
    // Keep this order: postcard encodes by declaration order and the fixture pins Floating = 0.
    #[default]
    Floating,
    Grounded,
}

/// Which form a component's signature left its matrix in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatrixForm {
    /// Folded to a Z-matrix: the minor reduces directly.
    Laplacian,
    /// Kept signed; a Gremban cover is built at factor time so the stored matrix never doubles.
    SignedPendingCover,
}

/// Map between original and SDDM coordinates; `Canonical` negates the col block, not identity.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) enum CoordinateMap {
    // Keep this order: postcard encodes by declaration order and the fixture pins Canonical = 0.
    #[default]
    Canonical,
    /// Diagonal congruence factors (sign · scale), leaving a frustrated component signed.
    Scaled(Box<[f64]>),
}

impl CoordinateMap {
    /// `values` spans the internal system, split at `n_eliminated`.
    pub(crate) fn fold(&self, values: &mut [f64], n_eliminated: usize) {
        match self {
            CoordinateMap::Canonical => {
                for value in &mut values[n_eliminated..] {
                    *value = -*value;
                }
            }
            CoordinateMap::Scaled(factors) => {
                debug_assert_eq!(values.len(), factors.len());
                for (value, &factor) in values.iter_mut().zip(factors.iter()) {
                    *value *= factor;
                }
            }
        }
    }

    /// The diagonal maps are involutions up to scale, so they re-apply `fold`.
    pub(crate) fn unfold(&self, values: &mut [f64], n_eliminated: usize) {
        self.fold(values, n_eliminated);
    }
}

/// Orient so the larger block is eliminated; arrays arrive `[rows | cols]`, leave eliminated-major.
pub(crate) fn orient_for_elimination(
    cross_tab: CrossTab,
    mut diagonal: Vec<f64>,
    mut globals: Vec<u32>,
) -> (CrossTab, Vec<f64>, Vec<u32>) {
    debug_assert_eq!(globals.len(), cross_tab.n_local());
    debug_assert_eq!(diagonal.len(), cross_tab.n_local());
    if cross_tab.n_cols() <= cross_tab.n_rows() {
        return (cross_tab, diagonal, globals);
    }
    diagonal.rotate_left(cross_tab.n_rows());
    globals.rotate_left(cross_tab.n_rows());
    (
        CrossTab {
            c: cross_tab.ct,
            ct: cross_tab.c,
        },
        diagonal,
        globals,
    )
}

/// Bipartite SDDM in eliminated-major form; arrays are flat, split at `n_eliminated`.
#[derive(Clone)]
pub(crate) struct SddmMatrix {
    pub(crate) cross_tab: CrossTab,
    pub(crate) diagonal: Vec<f64>,
    pub(crate) ground_edges: Vec<f64>,
    pub(crate) grounding: Grounding,
}

impl SddmMatrix {
    pub(crate) fn n_eliminated(&self) -> usize {
        self.cross_tab.n_rows()
    }

    pub(crate) fn n_kept(&self) -> usize {
        self.cross_tab.n_cols()
    }

    pub(crate) fn diagonal_kept(&self) -> &[f64] {
        &self.diagonal[self.n_eliminated()..]
    }

    pub(crate) fn surplus_eliminated(&self) -> &[f64] {
        &self.ground_edges[..self.n_eliminated()]
    }

    pub(crate) fn surplus_kept(&self) -> &[f64] {
        &self.ground_edges[self.n_eliminated()..]
    }
}

/// A connected component of a channel pair's cross-tab, converted to SDDM form.
#[derive(Clone)]
pub(crate) struct LocalComponent {
    pub(crate) matrix: SddmMatrix,
    pub(crate) form: MatrixForm,
    pub(crate) coordinates: CoordinateMap,
}

/// No diagonal scaling reaches weak dominance (Boman), so no SDDM form exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NotScalable;

/// Evidence a dominance scaling exceeded tolerance or budget under [`ScalingFailure::Warn`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct UncertifiedScaling {
    pub(crate) sweeps: usize,
    pub(crate) violation: f64,
}

pub(super) fn convert(
    cross_tab: CrossTab,
    diagonal: Vec<f64>,
    class: ComponentClass,
    scaling: &ScalingConfig,
) -> Result<(LocalComponent, Option<UncertifiedScaling>), NotScalable> {
    match class {
        ComponentClass::KnownLaplacian => {
            convert_known_laplacian(cross_tab, diagonal).map(|component| (component, None))
        }
        ComponentClass::General => convert_general(cross_tab, diagonal, scaling),
    }
}

/// Skipping the signed machinery keeps the dominant plain path at two streaming passes.
fn convert_known_laplacian(
    cross_tab: CrossTab,
    diagonal: Vec<f64>,
) -> Result<LocalComponent, NotScalable> {
    let row_sums = adjacency_sums(&cross_tab);
    let mut total_diagonal = 0.0;
    let mut total_mismatch = 0.0;
    for (&entry, &row_sum) in diagonal.iter().zip(row_sums.iter()) {
        total_diagonal += entry;
        total_mismatch += (entry - row_sum).abs();
    }
    if total_mismatch > LAPLACIAN_VALIDATION_BUDGET.tolerance(cross_tab.n_local(), total_diagonal) {
        return Err(NotScalable);
    }
    Ok(LocalComponent {
        matrix: SddmMatrix {
            ground_edges: vec![0.0; cross_tab.n_local()],
            cross_tab,
            diagonal: row_sums,
            grounding: Grounding::Floating,
        },
        form: MatrixForm::Laplacian,
        coordinates: CoordinateMap::default(),
    })
}

fn convert_general(
    cross_tab: CrossTab,
    diagonal: Vec<f64>,
    scaling: &ScalingConfig,
) -> Result<(LocalComponent, Option<UncertifiedScaling>), NotScalable> {
    let signs = folding_signs(&cross_tab);
    let relaxation = dominance_scaling(&cross_tab, &diagonal, scaling)?;
    if relaxation.violation > scaling.tolerance && scaling.on_failure == ScalingFailure::Error {
        return Err(NotScalable);
    }
    let (component, clamped_deficit) = match signs {
        Some(signs) => {
            let factors: Vec<f64> = signs
                .iter()
                .zip(relaxation.scales.iter())
                .map(|(&sign, &scale)| sign * scale)
                .collect();
            assemble(cross_tab, diagonal, factors, MatrixForm::Laplacian, scaling)?
        }
        None => {
            let n_rows = cross_tab.n_rows();
            let mut factors = relaxation.scales;
            for factor in &mut factors[n_rows..] {
                *factor = -*factor;
            }
            assemble(
                cross_tab,
                diagonal,
                factors,
                MatrixForm::SignedPendingCover,
                scaling,
            )?
        }
    };
    let violation = relaxation.violation.max(clamped_deficit);
    let uncertified = (violation > scaling.tolerance).then_some(UncertifiedScaling {
        sweeps: relaxation.sweeps,
        violation,
    });
    Ok((component, uncertified))
}

/// A [`MatrixForm::Laplacian`] must fold to a Z-matrix, so a positive off-diagonal is an error.
fn assemble(
    mut cross_tab: CrossTab,
    diagonal: Vec<f64>,
    factors: Vec<f64>,
    form: MatrixForm,
    scaling: &ScalingConfig,
) -> Result<(LocalComponent, f64), NotScalable> {
    let n_rows = cross_tab.n_rows();
    let enforce_z = form == MatrixForm::Laplacian;
    for i in 0..n_rows {
        let start = cross_tab.c.indptr[i] as usize;
        let end = cross_tab.c.indptr[i + 1] as usize;
        let columns = &cross_tab.c.indices[start..end];
        for (&j, value) in columns.iter().zip(&mut cross_tab.c.data[start..end]) {
            let folded = -factors[i] * factors[n_rows + j as usize] * *value;
            if !folded.is_finite() || (enforce_z && folded < 0.0) {
                return Err(NotScalable);
            }
            *value = folded;
        }
    }
    cross_tab.ct = cross_tab.c.transpose();

    let scaled_diagonal: Vec<f64> = diagonal
        .iter()
        .zip(factors.iter())
        .map(|(&entry, &factor)| entry * factor * factor)
        .collect();

    // Canonical bipartite factors need no stored map: `fold` applies that sign flip by default.
    let canonical = factors
        .iter()
        .enumerate()
        .all(|(i, &factor)| factor == if i < n_rows { 1.0 } else { -1.0 });
    let coordinates = if canonical {
        CoordinateMap::Canonical
    } else {
        CoordinateMap::Scaled(factors.into_boxed_slice())
    };

    let row_sums = magnitude_sums(&cross_tab);
    finalize(
        cross_tab,
        scaled_diagonal,
        row_sums,
        coordinates,
        form,
        scaling,
    )
}

/// Clamp roundoff deficits, retain surplus as ground edges, and classify the [`Grounding`].
fn finalize(
    cross_tab: CrossTab,
    mut scaled_diagonal: Vec<f64>,
    row_sums: Vec<f64>,
    coordinates: CoordinateMap,
    form: MatrixForm,
    scaling: &ScalingConfig,
) -> Result<(LocalComponent, f64), NotScalable> {
    let n = cross_tab.n_local();
    let mut ground_edges = vec![0.0; n];

    let mut total_diagonal = 0.0;
    let mut total_surplus = 0.0;
    let mut clamped_deficit = 0.0f64;
    for ((diagonal, &row_sum), surplus) in scaled_diagonal
        .iter_mut()
        .zip(row_sums.iter())
        .zip(ground_edges.iter_mut())
    {
        if !diagonal.is_finite() || *diagonal <= 0.0 {
            return Err(NotScalable);
        }
        let deficit = (row_sum - *diagonal) / *diagonal;
        if deficit > scaling.tolerance {
            if scaling.on_failure == ScalingFailure::Error {
                return Err(NotScalable);
            }
            clamped_deficit = clamped_deficit.max(deficit);
        }
        *diagonal = diagonal.max(row_sum);
        *surplus = (*diagonal - row_sum).max(0.0);
        total_diagonal += *diagonal;
        total_surplus += *surplus;
    }

    let floats = total_surplus <= FLOATING_CLASSIFICATION_BUDGET.tolerance(n, total_diagonal);
    if floats {
        scaled_diagonal = row_sums;
        ground_edges.fill(0.0);
    }
    let grounding = if floats {
        Grounding::Floating
    } else {
        Grounding::Grounded
    };
    Ok((
        LocalComponent {
            matrix: SddmMatrix {
                cross_tab,
                diagonal: scaled_diagonal,
                ground_edges,
                grounding,
            },
            form,
            coordinates,
        },
        clamped_deficit,
    ))
}

/// `None` when the component is frustrated: a negative cycle admits no signature.
fn folding_signs(cross_tab: &CrossTab) -> Option<Vec<f64>> {
    let n = cross_tab.n_local();
    let mut signs = vec![0.0; n];
    if n == 0 {
        return Some(signs);
    }
    for root in 0..n {
        if signs[root] != 0.0 {
            continue;
        }
        signs[root] = 1.0;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            for (j, value) in cross_tab.neighbors(i) {
                let expected = -signs[i] * value.signum();
                if signs[j] == 0.0 {
                    signs[j] = expected;
                    stack.push(j);
                } else if signs[j] != expected {
                    return None;
                }
            }
        }
    }
    Some(signs)
}

struct DominanceScaling {
    scales: Vec<f64>,
    sweeps: usize,
    /// Largest relative dominance violation at hand-over, clamped at 0.
    violation: f64,
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
        if !normalize_positive(&mut mu) {
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

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&x, &y)| x * y).sum()
}

fn normalize_positive(values: &mut [f64]) -> bool {
    let peak = values.iter().copied().fold(0.0f64, f64::max);
    if !peak.is_finite() || peak <= 0.0 {
        return false;
    }
    for value in values.iter_mut() {
        *value /= peak;
    }
    values.iter().all(|value| value.is_finite() && *value > 0.0)
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
                if violation <= scaling.tolerance {
                    return Ok(ScalingResult {
                        candidate: operator.materialize_candidate(
                            &boundary_small,
                            &large_work,
                            violation,
                        )?,
                        iterations: iteration,
                    });
                }
            }
            return Ok(ScalingResult {
                candidate: best,
                iterations: iteration,
            });
        }

        if let Some(violation) = operator.candidate_violation(
            &cg.solution,
            CandidateKind::Strict,
            &mut large_work,
            &mut small_image,
        ) {
            if violation < best.violation {
                best = operator.materialize_candidate(&cg.solution, &large_work, violation)?;
            }
            if violation <= scaling.tolerance {
                return Ok(ScalingResult {
                    candidate: best,
                    iterations: iteration,
                });
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

fn dominance_scaling(
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

fn adjacency_sums(cross_tab: &CrossTab) -> Vec<f64> {
    (0..cross_tab.n_local())
        .map(|i| cross_tab.neighbors(i).map(|(_, v)| v).sum())
        .collect()
}

fn magnitude_sums(cross_tab: &CrossTab) -> Vec<f64> {
    (0..cross_tab.n_local())
        .map(|i| cross_tab.neighbors(i).map(|(_, v)| v.abs()).sum())
        .collect()
}

#[cfg(test)]
mod tests;
