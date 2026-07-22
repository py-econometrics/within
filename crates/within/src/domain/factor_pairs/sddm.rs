//! Validated conversion from a signed bipartite Gram block to SDDM form.
//!
//! One signed diagonal coordinate map `S` makes `S A S` a Z-matrix and scales
//! it to diagonal dominance. Surplus is retained as explicit ground-edge
//! weight, so downstream reduction never has to infer rank or grounding from
//! rounded diagonals. Plain components take a fast path: their Laplacian claim
//! is validated and adopted without sign folding, scaling, or a data rewrite.
//! Frustrated components — where no signature exists — keep their single
//! *signed* operator (dominance-scaled but not folded to a Z-matrix) and carry
//! a [`Reduction::Cover`] marker; the Gremban double cover that makes the
//! reduced Schur SDDM is built transiently at factor time (see
//! [`crate::block_elim`]), so the stored operator stays single-sized.

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SolveSpace {
    // Keep this order: postcard encodes enum discriminants by declaration
    // order, and the wire fixture pins Floating = 0; new variants append last.
    #[default]
    Floating,
    Grounded,
    /// Signed operator whose reduced Schur self-grounds through its Gremban
    /// cover ([`ReductionKind::Cover`]): the antisymmetric `[b, -b]` embed
    /// balances the RHS, so the operator-level solve does no mean-subtraction
    /// and injects no ground current.
    Signed,
}

/// The reduction strategy chosen from a component's signature, before its
/// grounding is known: `Direct` folds to a Z-matrix and reduces its minor
/// straight; `Cover` keeps the signed operator and defers a Gremban double
/// cover to factor time so the stored operator never doubles. Orthogonal to
/// [`CoordinateMap`] (an operator-level congruence). Resolved into a
/// [`Reduction`] once the grounding is classified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReductionKind {
    Direct,
    Cover,
}

/// Grounding of a directly-reduced SDDM minor: `Floating` anchors one node,
/// `Grounded` factors the full complement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Grounding {
    Floating,
    Grounded,
}

impl Grounding {
    /// The solve space a floating/grounded minor gauges.
    pub(crate) fn solve_space(self) -> SolveSpace {
        match self {
            Grounding::Floating => SolveSpace::Floating,
            Grounding::Grounded => SolveSpace::Grounded,
        }
    }
}

/// Resolved reduction state of a local component: the [`ReductionKind`]
/// strategy paired with the solve space it produced. Only these combinations
/// are reachable, so the illegal signed-with-direct pairing cannot be
/// constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reduction {
    /// Direct Schur factorization of a floating or grounded minor.
    Direct(Grounding),
    /// Gremban-cover reduction of a signed operator; solves in [`SolveSpace::Signed`].
    Cover,
}

impl Reduction {
    /// The solve space this reduction gauges.
    pub(crate) fn solve_space(self) -> SolveSpace {
        match self {
            Reduction::Direct(grounding) => grounding.solve_space(),
            Reduction::Cover => SolveSpace::Signed,
        }
    }
}

/// Map between original and SDDM coordinates, applied to vectors at the
/// solve boundary. `Canonical` is the bipartite map (negate the `r` block) —
/// not the identity.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) enum CoordinateMap {
    // Keep this order: postcard encodes enum discriminants by declaration
    // order, and the wire fixture pins Canonical = 0.
    #[default]
    Canonical,
    /// Diagonal congruence factors (sign · scale). For a frustrated component
    /// these are the canonical bipartite signs (`+` on q, `−` on r) times the
    /// dominance scaling, leaving the operator signed.
    Scaled(Box<[f64]>),
}

#[derive(Clone)]
pub(crate) struct GroundEdges {
    pub(crate) q: Vec<f64>,
    pub(crate) r: Vec<f64>,
}

impl CoordinateMap {
    /// Map an original-coordinate RHS into SDDM coordinates. `values` spans
    /// the internal system and `n_q` is its q-size.
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
        }
    }

    /// Map an SDDM-coordinate solution back to original coordinates; the
    /// diagonal maps are involutions up to scale, so they re-apply `fold`.
    pub(crate) fn unfold(&self, values: &mut [f64], n_q: usize) {
        self.fold(values, n_q);
    }
}

#[derive(Clone)]
pub(crate) struct LocalComponent {
    pub(crate) cross_tab: CrossTab,
    pub(crate) diagonals: BlockDiagonals,
    pub(crate) ground_edges: GroundEdges,
    pub(crate) coordinates: CoordinateMap,
    pub(crate) reduction: Reduction,
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
) -> Result<(LocalComponent, Option<UncertifiedScaling>), NotScalable> {
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
) -> Result<LocalComponent, NotScalable> {
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
    Ok(LocalComponent {
        cross_tab,
        diagonals: row_sums,
        ground_edges,
        coordinates: CoordinateMap::default(),
        reduction: Reduction::Direct(Grounding::Floating),
    })
}

fn convert_general(
    cross_tab: CrossTab,
    diagonals: BlockDiagonals,
    scaling: &ScalingConfig,
) -> Result<(LocalComponent, Option<UncertifiedScaling>), NotScalable> {
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
            assemble(
                cross_tab,
                diagonals,
                factors,
                ReductionKind::Direct,
                scaling,
            )?
        }
        None => {
            let n_q = cross_tab.n_q();
            let mut factors = relaxation.scales;
            for factor in &mut factors[n_q..] {
                *factor = -*factor;
            }
            assemble(cross_tab, diagonals, factors, ReductionKind::Cover, scaling)?
        }
    };
    let violation = relaxation.violation.max(clamped_deficit);
    let uncertified = (violation > scaling.tolerance).then_some(UncertifiedScaling {
        sweeps: relaxation.sweeps,
        violation,
    });
    Ok((component, uncertified))
}

/// Fold the component through `factors` and assemble the validated SDDM form.
/// The `reduction` fixes the fold contract: a [`ReductionKind::Direct`]
/// component must fold to a true Z-matrix, so any off-diagonal the signature
/// failed to drive nonnegative is an error; a [`ReductionKind::Cover`] component
/// keeps its single *signed* operator (negatives retained) and grounds through a
/// Gremban double cover built at factor time (see [`crate::block_elim`]).
/// Dominance is classified against *magnitude* row sums either way — after a
/// Z-fold every off-diagonal is nonnegative, so magnitudes match the signed
/// adjacency there. Returns the largest relative diagonal deficit that was
/// clamped; deficits beyond tolerance are errors under [`ScalingFailure::Error`].
fn assemble(
    mut cross_tab: CrossTab,
    diagonals: BlockDiagonals,
    factors: Vec<f64>,
    reduction: ReductionKind,
    scaling: &ScalingConfig,
) -> Result<(LocalComponent, f64), NotScalable> {
    let n_q = cross_tab.n_q();
    let enforce_z = reduction == ReductionKind::Direct;
    for i in 0..n_q {
        let start = cross_tab.c.indptr[i] as usize;
        let end = cross_tab.c.indptr[i + 1] as usize;
        let columns = &cross_tab.c.indices[start..end];
        for (&j, value) in columns.iter().zip(&mut cross_tab.c.data[start..end]) {
            let folded = -factors[i] * factors[n_q + j as usize] * *value;
            if !folded.is_finite() || (enforce_z && folded < 0.0) {
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

    // Canonical bipartite factors (`+1` on q, `−1` on r) need no stored map: the
    // congruence is just the sign flip `fold` applies by default. This holds for
    // a plain already-dominant component and equally for a frustrated one, so it
    // is detected regardless of the reduction.
    let canonical = factors
        .iter()
        .enumerate()
        .all(|(i, &factor)| factor == if i < n_q { 1.0 } else { -1.0 });
    let coordinates = if canonical {
        CoordinateMap::Canonical
    } else {
        CoordinateMap::Scaled(factors.into_boxed_slice())
    };

    let magnitude_sums = |block: &CsrBlock| -> Vec<f64> {
        (0..block.nrows)
            .map(|i| block.row(i).map(|(_, v)| v.abs()).sum())
            .collect()
    };
    let row_sums = BlockDiagonals {
        q: magnitude_sums(&cross_tab.c),
        r: magnitude_sums(&cross_tab.ct),
    };

    finalize(
        cross_tab,
        scaled_diagonals,
        row_sums,
        coordinates,
        reduction,
        scaling,
    )
}

/// Validate weak dominance of an assembled operator against `row_sums` (signed
/// adjacency for a Z-matrix, magnitudes for a signed operator): clamp roundoff
/// deficits, retain surplus as ground edges, and resolve the [`Reduction`]
/// state. A `Cover`-reduced component is always [`SolveSpace::Signed`]; the
/// float/ground classification only decides whether its surplus is retained
/// (the cover self-grounds either way). Returns the largest relative deficit
/// that was clamped; deficits beyond tolerance are errors under
/// [`ScalingFailure::Error`].
fn finalize(
    cross_tab: CrossTab,
    mut scaled_diagonals: BlockDiagonals,
    row_sums: BlockDiagonals,
    coordinates: CoordinateMap,
    reduction: ReductionKind,
    scaling: &ScalingConfig,
) -> Result<(LocalComponent, f64), NotScalable> {
    let n = cross_tab.n_local();
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

    let floats = total_surplus <= FLOATING_CLASSIFICATION_BUDGET.tolerance(n, total_diagonal);
    if floats {
        scaled_diagonals = row_sums;
        ground_edges.q.fill(0.0);
        ground_edges.r.fill(0.0);
    }
    let reduction = match reduction {
        ReductionKind::Cover => Reduction::Cover,
        ReductionKind::Direct if floats => Reduction::Direct(Grounding::Floating),
        ReductionKind::Direct => Reduction::Direct(Grounding::Grounded),
    };

    Ok((
        LocalComponent {
            cross_tab,
            diagonals: scaled_diagonals,
            ground_edges,
            coordinates,
            reduction,
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
