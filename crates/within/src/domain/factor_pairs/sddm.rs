//! Validated conversion from a signed bipartite Gram block to SDDM form.
//!
//! One signed diagonal coordinate map `S` makes `S A S` a Z-matrix and scales
//! it to diagonal dominance. Surplus is retained as explicit ground-edge
//! weight, so downstream reduction never has to infer rank or grounding from
//! rounded diagonals. Plain components take a fast path: their Laplacian claim
//! is validated and adopted without sign folding, scaling, or a data rewrite.

use super::ComponentClass;
use crate::config::{ScalingConfig, ScalingFailure};
use crate::domain::{BlockDiagonals, CrossTab};

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
// A false Grounded classification retains noise-sized edges; a false Floating
// classification deletes an identified direction.
const FLOATING_CLASSIFICATION_BUDGET: RoundoffBudget = RoundoffBudget { ulps: 4.0 };

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SolveSpace {
    #[default]
    Floating,
    Grounded,
}

/// Diagonal map between original and SDDM coordinates, applied to vectors at
/// the solve boundary. `None` is the canonical bipartite map (negate the `r`
/// block) — not the identity.
#[derive(Clone, Debug, Default)]
pub(crate) struct CoordinateMap {
    factors: Option<Box<[f64]>>,
}

#[derive(Clone)]
pub(crate) struct GroundEdges {
    pub(crate) q: Vec<f64>,
    pub(crate) r: Vec<f64>,
}

impl CoordinateMap {
    pub(crate) fn apply(&self, values: &mut [f64], n_q: usize) {
        match &self.factors {
            Some(factors) => {
                debug_assert_eq!(values.len(), factors.len());
                for (value, &factor) in values.iter_mut().zip(factors.iter()) {
                    *value *= factor;
                }
            }
            None => {
                for value in &mut values[n_q..] {
                    *value = -*value;
                }
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct SddmComponent {
    pub(crate) cross_tab: CrossTab,
    pub(crate) diagonals: BlockDiagonals,
    pub(crate) ground_edges: GroundEdges,
    pub(crate) coordinates: CoordinateMap,
    pub(crate) solve_space: SolveSpace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConversionError {
    Frustrated,
    NotScalable,
}

/// Evidence that a component's dominance scaling exceeded tolerance or budget
/// under [`ScalingFailure::Warn`]; the caller attaches pair context and
/// surfaces it as a [`crate::BuildWarning`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct UncertifiedScaling {
    pub(crate) sweeps: usize,
    pub(crate) violation: f64,
}

pub(super) fn convert(
    cross_tab: CrossTab,
    diagonals: BlockDiagonals,
    class: ComponentClass,
    scaling: &ScalingConfig,
) -> Result<(SddmComponent, Option<UncertifiedScaling>), ConversionError> {
    match class {
        ComponentClass::KnownLaplacian => {
            convert_known_laplacian(cross_tab, diagonals).map(|component| (component, None))
        }
        ComponentClass::General => convert_general(cross_tab, diagonals, scaling),
    }
}

/// Plain components are Laplacian by construction: validate that claim and
/// adopt the canonical SDDM form. Skipping the signed machinery keeps the
/// dominant plain path at two streaming passes over the cross data.
fn convert_known_laplacian(
    cross_tab: CrossTab,
    diagonals: BlockDiagonals,
) -> Result<SddmComponent, ConversionError> {
    let row_sums = adjacency_sums(&cross_tab);
    let mut total_diagonal = 0.0;
    let mut total_mismatch = 0.0;
    for (&diagonal, &row_sum) in diagonals
        .q
        .iter()
        .zip(row_sums.q.iter())
        .chain(diagonals.r.iter().zip(row_sums.r.iter()))
    {
        total_diagonal += diagonal;
        total_mismatch += (diagonal - row_sum).abs();
    }
    if total_mismatch > LAPLACIAN_VALIDATION_BUDGET.tolerance(cross_tab.n_local(), total_diagonal) {
        return Err(ConversionError::NotScalable);
    }
    let ground_edges = GroundEdges {
        q: vec![0.0; cross_tab.n_q()],
        r: vec![0.0; cross_tab.n_r()],
    };
    Ok(SddmComponent {
        cross_tab,
        diagonals: row_sums,
        ground_edges,
        coordinates: CoordinateMap::default(),
        solve_space: SolveSpace::Floating,
    })
}

fn convert_general(
    cross_tab: CrossTab,
    diagonals: BlockDiagonals,
    scaling: &ScalingConfig,
) -> Result<(SddmComponent, Option<UncertifiedScaling>), ConversionError> {
    let signs = folding_signs(&cross_tab)?;
    let relaxation = dominance_scaling(&cross_tab, &diagonals, scaling)?;
    if !relaxation.certified && scaling.on_failure == ScalingFailure::Error {
        return Err(ConversionError::NotScalable);
    }
    let factors: Vec<f64> = signs
        .iter()
        .zip(relaxation.scales.iter())
        .map(|(&sign, &scale)| sign * scale)
        .collect();
    let (component, clamped_deficit) = assemble(cross_tab, diagonals, factors, scaling)?;
    let violation = relaxation.violation.max(clamped_deficit);
    let uncertified = (violation > scaling.tolerance).then_some(UncertifiedScaling {
        sweeps: relaxation.sweeps,
        violation,
    });
    Ok((component, uncertified))
}

/// Fold the component through `factors` and assemble the validated SDDM form.
/// Returns the largest relative diagonal deficit that was clamped; deficits
/// beyond tolerance are errors under [`ScalingFailure::Error`].
fn assemble(
    mut cross_tab: CrossTab,
    diagonals: BlockDiagonals,
    factors: Vec<f64>,
    scaling: &ScalingConfig,
) -> Result<(SddmComponent, f64), ConversionError> {
    let n_q = cross_tab.n_q();
    let n = cross_tab.n_local();
    for i in 0..n_q {
        let start = cross_tab.c.indptr[i] as usize;
        let end = cross_tab.c.indptr[i + 1] as usize;
        let columns = &cross_tab.c.indices[start..end];
        for (&j, value) in columns.iter().zip(&mut cross_tab.c.data[start..end]) {
            let folded = -factors[i] * factors[n_q + j as usize] * *value;
            if !folded.is_finite() || folded < 0.0 {
                return Err(ConversionError::NotScalable);
            }
            *value = folded;
        }
    }
    cross_tab.ct = cross_tab.c.transpose();

    let mut scaled_diagonals = BlockDiagonals {
        q: diagonals
            .q
            .iter()
            .zip(factors[..n_q].iter())
            .map(|(&diagonal, &factor)| diagonal * factor * factor)
            .collect(),
        r: diagonals
            .r
            .iter()
            .zip(factors[n_q..].iter())
            .map(|(&diagonal, &factor)| diagonal * factor * factor)
            .collect(),
    };
    let row_sums = adjacency_sums(&cross_tab);
    let mut ground_edges = GroundEdges {
        q: vec![0.0; n_q],
        r: vec![0.0; cross_tab.n_r()],
    };

    let mut total_diagonal = 0.0;
    let mut total_surplus = 0.0;
    let mut clamped_deficit = 0.0f64;
    for ((diagonal, row_sum), row_surplus) in scaled_diagonals
        .q
        .iter_mut()
        .zip(row_sums.q.iter().copied())
        .zip(ground_edges.q.iter_mut())
        .chain(
            scaled_diagonals
                .r
                .iter_mut()
                .zip(row_sums.r.iter().copied())
                .zip(ground_edges.r.iter_mut()),
        )
    {
        if !diagonal.is_finite() || *diagonal <= 0.0 {
            return Err(ConversionError::NotScalable);
        }
        let deficit = (row_sum - *diagonal) / diagonal.max(row_sum);
        if deficit > scaling.tolerance {
            if scaling.on_failure == ScalingFailure::Error {
                return Err(ConversionError::NotScalable);
            }
            clamped_deficit = clamped_deficit.max(deficit);
        }
        *diagonal = diagonal.max(row_sum);
        *row_surplus = (*diagonal - row_sum).max(0.0);
        total_diagonal += *diagonal;
        total_surplus += *row_surplus;
    }

    let solve_space =
        if total_surplus <= FLOATING_CLASSIFICATION_BUDGET.tolerance(n, total_diagonal) {
            scaled_diagonals = row_sums;
            ground_edges.q.fill(0.0);
            ground_edges.r.fill(0.0);
            SolveSpace::Floating
        } else {
            SolveSpace::Grounded
        };

    let canonical = factors
        .iter()
        .enumerate()
        .all(|(i, &factor)| factor == if i < n_q { 1.0 } else { -1.0 });
    let coordinates = CoordinateMap {
        factors: (!canonical).then(|| factors.into_boxed_slice()),
    };

    Ok((
        SddmComponent {
            cross_tab,
            diagonals: scaled_diagonals,
            ground_edges,
            coordinates,
            solve_space,
        },
        clamped_deficit,
    ))
}

fn folding_signs(cross_tab: &CrossTab) -> Result<Vec<f64>, ConversionError> {
    let n = cross_tab.n_local();
    let mut signs = vec![0.0; n];
    if n == 0 {
        return Ok(signs);
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
                    return Err(ConversionError::Frustrated);
                }
            }
        }
    }
    Ok(signs)
}

struct DominanceScaling {
    scales: Vec<f64>,
    certified: bool,
    sweeps: usize,
    /// Largest relative dominance violation at hand-over (0 when certified).
    violation: f64,
}

fn dominance_scaling(
    cross_tab: &CrossTab,
    diagonals: &BlockDiagonals,
    scaling: &ScalingConfig,
) -> Result<DominanceScaling, ConversionError> {
    let n_q = cross_tab.n_q();
    let n = cross_tab.n_local();
    let diagonal = |i: usize| {
        if i < n_q {
            diagonals.q[i]
        } else {
            diagonals.r[i - n_q]
        }
    };
    if (0..n).any(|i| !diagonal(i).is_finite() || diagonal(i) <= 0.0) {
        return Err(ConversionError::NotScalable);
    }

    let already_dominant = (0..n).all(|i| {
        let row_sum: f64 = cross_tab.neighbors(i).map(|(_, value)| value.abs()).sum();
        row_sum <= diagonal(i) * (1.0 + scaling.tolerance)
    });
    if already_dominant {
        return Ok(DominanceScaling {
            scales: vec![1.0; n],
            certified: true,
            sweeps: 0,
            violation: 0.0,
        });
    }

    let inv_sqrt: Vec<f64> = (0..n).map(|i| 1.0 / diagonal(i).sqrt()).collect();
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

    let mut mu = vec![1.0; n];
    let mut violation = violation_of(&mu);
    let mut sweeps = 0;
    while violation > scaling.tolerance && sweeps < scaling.max_sweeps {
        for i in 0..n {
            let target: f64 = cross_tab
                .neighbors(i)
                .map(|(j, value)| value.abs() * inv_sqrt[i] * inv_sqrt[j] * mu[j])
                .sum();
            if target > mu[i] {
                mu[i] = target;
            }
        }
        // The certificate is invariant under a global rescale of μ; normalizing
        // keeps a non-scalable component's geometric growth finite so budget
        // exhaustion hands over a usable best-effort scaling.
        let peak = mu.iter().copied().fold(0.0f64, f64::max);
        if !peak.is_finite() {
            return Err(ConversionError::NotScalable);
        }
        for value in &mut mu {
            *value /= peak;
        }
        violation = violation_of(&mu);
        sweeps += 1;
    }

    let scales: Vec<f64> = mu
        .iter()
        .zip(inv_sqrt.iter())
        .map(|(&value, &normalizer)| value * normalizer)
        .collect();
    if !scales.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err(ConversionError::NotScalable);
    }
    Ok(DominanceScaling {
        scales,
        certified: violation <= scaling.tolerance,
        sweeps,
        violation: violation.max(0.0),
    })
}

fn adjacency_sums(cross_tab: &CrossTab) -> BlockDiagonals {
    let sum_rows = |block: &crate::csr_block::CsrBlock| {
        (0..block.nrows)
            .map(|i| block.row(i).map(|(_, v)| v).sum())
            .collect()
    };
    BlockDiagonals {
        q: sum_rows(&cross_tab.c),
        r: sum_rows(&cross_tab.ct),
    }
}

#[cfg(test)]
mod tests;
