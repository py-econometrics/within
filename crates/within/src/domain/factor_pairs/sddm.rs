//! Validated conversion from a signed bipartite Gram block to SDDM form.
//!
//! One signed diagonal coordinate map `S` makes `S A S` a Z-matrix and scales
//! it to diagonal dominance. Surplus is retained as explicit ground-edge
//! weight, so downstream reduction never has to infer rank or grounding from
//! rounded diagonals. Plain components take a fast path: their Laplacian claim
//! is validated and adopted without sign folding, scaling, or a data rewrite.
//! Frustrated components — where no signature exists — convert through their
//! Gremban double cover, which is SDDM under the same dominance scaling and
//! acts on the antisymmetric subspace as the scaled original.

use super::ComponentClass;
use crate::config::{ScalingConfig, ScalingFailure};
use crate::csr_block::CsrBlock;
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

/// Map between original and SDDM coordinates, applied to vectors at the
/// solve boundary. `Canonical` is the bipartite map (negate the `r` block) —
/// not the identity.
#[derive(Clone, Debug, Default)]
pub(crate) enum CoordinateMap {
    #[default]
    Canonical,
    /// Diagonal congruence factors (sign · scale) of a balanced component.
    Scaled(Box<[f64]>),
    /// Gremban double cover of a frustrated component: `factors` (canonical
    /// sign · scale, external length) map the original coordinates, and
    /// vectors pass through the doubled system via the antisymmetric
    /// embedding `[z, -z]` and read-back `(x⁺ - x⁻) / 2`.
    Cover(Box<[f64]>),
}

#[derive(Clone)]
pub(crate) struct GroundEdges {
    pub(crate) q: Vec<f64>,
    pub(crate) r: Vec<f64>,
}

impl CoordinateMap {
    /// DOF count exposed to gather/scatter; a cover solves an internally
    /// doubled system.
    pub(crate) fn n_external(&self, n_internal: usize) -> usize {
        match self {
            CoordinateMap::Cover(_) => n_internal / 2,
            _ => n_internal,
        }
    }

    /// Map an original-coordinate RHS into SDDM coordinates. `values` spans
    /// the internal system and `n_q` is its internal q-size; for covers only
    /// the first `n_external` entries hold data on entry.
    pub(crate) fn fold(&self, values: &mut [f64], n_q: usize) {
        match self {
            CoordinateMap::Canonical => {
                for value in &mut values[n_q..] {
                    *value = -*value;
                }
            }
            CoordinateMap::Scaled(factors) => {
                debug_assert_eq!(values.len(), factors.len());
                for (value, &factor) in values.iter_mut().zip(factors.iter()) {
                    *value *= factor;
                }
            }
            CoordinateMap::Cover(factors) => {
                let (n_ext, n_q) = (values.len() / 2, n_q / 2);
                let n_r = n_ext - n_q;
                debug_assert_eq!(n_ext, factors.len());
                for (value, &factor) in values[..n_ext].iter_mut().zip(factors.iter()) {
                    *value *= factor;
                }
                // Expand [z_q | z_r] into [z_q | -z_q | z_r | -z_r]; block
                // moves ordered so no source is clobbered before its copy.
                values.copy_within(n_q..n_ext, 2 * n_q + n_r);
                values.copy_within(n_q..n_ext, 2 * n_q);
                values.copy_within(..n_q, n_q);
                for value in &mut values[n_q..2 * n_q] {
                    *value = -*value;
                }
                for value in &mut values[2 * n_q + n_r..] {
                    *value = -*value;
                }
            }
        }
    }

    /// Map an SDDM-coordinate solution back to original coordinates; the
    /// diagonal maps are involutions up to scale, so they re-apply `fold`.
    pub(crate) fn unfold(&self, values: &mut [f64], n_q: usize) {
        match self {
            CoordinateMap::Canonical | CoordinateMap::Scaled(_) => self.fold(values, n_q),
            CoordinateMap::Cover(factors) => {
                let (n_ext, n_q) = (values.len() / 2, n_q / 2);
                let n_r = n_ext - n_q;
                for i in 0..n_q {
                    values[i] = 0.5 * (values[i] - values[n_q + i]);
                }
                for i in 0..n_r {
                    values[n_q + i] = 0.5 * (values[2 * n_q + i] - values[2 * n_q + n_r + i]);
                }
                for (value, &factor) in values[..n_ext].iter_mut().zip(factors.iter()) {
                    *value *= factor;
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

/// The component admits no diagonal scaling to weak dominance (Boman); its
/// comparison matrix is not PSD, so no SDDM form — direct or covered — exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NotScalable;

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
) -> Result<(SddmComponent, Option<UncertifiedScaling>), NotScalable> {
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
) -> Result<SddmComponent, NotScalable> {
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
        return Err(NotScalable);
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
) -> Result<(SddmComponent, Option<UncertifiedScaling>), NotScalable> {
    let signs = folding_signs(&cross_tab);
    let relaxation = dominance_scaling(&cross_tab, &diagonals, scaling)?;
    if !relaxation.certified && scaling.on_failure == ScalingFailure::Error {
        return Err(NotScalable);
    }
    let (component, clamped_deficit) = match signs {
        Some(signs) => {
            let factors: Vec<f64> = signs
                .iter()
                .zip(relaxation.scales.iter())
                .map(|(&sign, &scale)| sign * scale)
                .collect();
            assemble(cross_tab, diagonals, factors, scaling)?
        }
        None => assemble_cover(cross_tab, diagonals, relaxation.scales, scaling)?,
    };
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
) -> Result<(SddmComponent, f64), NotScalable> {
    let n_q = cross_tab.n_q();
    for i in 0..n_q {
        let start = cross_tab.c.indptr[i] as usize;
        let end = cross_tab.c.indptr[i + 1] as usize;
        let columns = &cross_tab.c.indices[start..end];
        for (&j, value) in columns.iter().zip(&mut cross_tab.c.data[start..end]) {
            let folded = -factors[i] * factors[n_q + j as usize] * *value;
            if !folded.is_finite() || folded < 0.0 {
                return Err(NotScalable);
            }
            *value = folded;
        }
    }
    cross_tab.ct = cross_tab.c.transpose();

    let scaled_diagonals = BlockDiagonals {
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

    let canonical = factors
        .iter()
        .enumerate()
        .all(|(i, &factor)| factor == if i < n_q { 1.0 } else { -1.0 });
    let coordinates = if canonical {
        CoordinateMap::Canonical
    } else {
        CoordinateMap::Scaled(factors.into_boxed_slice())
    };
    finalize(cross_tab, scaled_diagonals, coordinates, scaling)
}

/// Assemble the Gremban double cover of a frustrated component: scaled cell
/// magnitudes, raw-nonnegative cells within each copy, raw-negative cells
/// across copies. The cover is SDDM exactly when `scales` certifies dominance
/// of the magnitudes, and acts on the antisymmetric subspace as the scaled
/// original.
fn assemble_cover(
    cross_tab: CrossTab,
    diagonals: BlockDiagonals,
    scales: Vec<f64>,
    scaling: &ScalingConfig,
) -> Result<(SddmComponent, f64), NotScalable> {
    let n_q = cross_tab.n_q();
    let n_r = cross_tab.n_r();

    let mut indptr = Vec::with_capacity(2 * n_q + 1);
    let mut indices = Vec::with_capacity(2 * cross_tab.c.nnz());
    let mut data = Vec::with_capacity(2 * cross_tab.c.nnz());
    indptr.push(0u32);
    for copy_shifted in [false, true] {
        for i in 0..n_q {
            let start = cross_tab.c.indptr[i] as usize;
            let end = cross_tab.c.indptr[i + 1] as usize;
            // Emit the unshifted column block before the shifted one so each
            // row's columns stay sorted.
            for column_shifted in [false, true] {
                let column_base = if column_shifted { n_r as u32 } else { 0 };
                let select_negative = column_shifted != copy_shifted;
                for idx in start..end {
                    let value = cross_tab.c.data[idx];
                    if (value < 0.0) != select_negative {
                        continue;
                    }
                    let scaled =
                        scales[i] * value * scales[n_q + cross_tab.c.indices[idx] as usize];
                    if !scaled.is_finite() {
                        return Err(NotScalable);
                    }
                    indices.push(cross_tab.c.indices[idx] + column_base);
                    data.push(scaled.abs());
                }
            }
            indptr.push(indices.len() as u32);
        }
    }
    let c = CsrBlock {
        indptr,
        indices,
        data,
        nrows: 2 * n_q,
        ncols: 2 * n_r,
    };
    let ct = c.transpose();

    let doubled = |diagonal: &[f64], scales: &[f64]| -> Vec<f64> {
        let scaled: Vec<f64> = diagonal
            .iter()
            .zip(scales.iter())
            .map(|(&diagonal, &scale)| diagonal * scale * scale)
            .collect();
        scaled.repeat(2)
    };
    let scaled_diagonals = BlockDiagonals {
        q: doubled(&diagonals.q, &scales[..n_q]),
        r: doubled(&diagonals.r, &scales[n_q..]),
    };

    let mut factors = scales;
    for factor in &mut factors[n_q..] {
        *factor = -*factor;
    }
    finalize(
        CrossTab { c, ct },
        scaled_diagonals,
        CoordinateMap::Cover(factors.into_boxed_slice()),
        scaling,
    )
}

/// Validate weak dominance of an assembled SDDM form: clamp roundoff
/// deficits, retain surplus as ground edges, and classify the solve space.
/// Returns the largest relative deficit that was clamped; deficits beyond
/// tolerance are errors under [`ScalingFailure::Error`].
fn finalize(
    cross_tab: CrossTab,
    mut scaled_diagonals: BlockDiagonals,
    coordinates: CoordinateMap,
    scaling: &ScalingConfig,
) -> Result<(SddmComponent, f64), NotScalable> {
    let n = cross_tab.n_local();
    let row_sums = adjacency_sums(&cross_tab);
    let mut ground_edges = GroundEdges {
        q: vec![0.0; cross_tab.n_q()],
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
            return Err(NotScalable);
        }
        let deficit = (row_sum - *diagonal) / diagonal.max(row_sum);
        if deficit > scaling.tolerance {
            if scaling.on_failure == ScalingFailure::Error {
                return Err(NotScalable);
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

/// A per-node signature folding every off-diagonal nonpositive, or `None`
/// when the component is frustrated (a negative cycle admits no signature).
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
    certified: bool,
    sweeps: usize,
    /// Largest relative dominance violation at hand-over (0 when certified).
    violation: f64,
}

fn dominance_scaling(
    cross_tab: &CrossTab,
    diagonals: &BlockDiagonals,
    scaling: &ScalingConfig,
) -> Result<DominanceScaling, NotScalable> {
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
        return Err(NotScalable);
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
            return Err(NotScalable);
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
        return Err(NotScalable);
    }
    Ok(DominanceScaling {
        scales,
        certified: violation <= scaling.tolerance,
        sweeps,
        violation: violation.max(0.0),
    })
}

fn adjacency_sums(cross_tab: &CrossTab) -> BlockDiagonals {
    let sum_rows = |block: &CsrBlock| {
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
