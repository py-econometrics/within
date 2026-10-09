//! A folded slope component's [`Grounding`], read from its observations, not its Gram.

use crate::domain::cross_tab::PairColumns;

use super::Grounding;

// 10u: a scaling that misses the kernel leaves surplus that floating would drop.
const SCALED_FLOATING_BOUND: f64 = 5.0 * f64::EPSILON;

/// `Floating` iff a tested component's scaling `f` has `fᵀGf ≤ 10u·fᵀDf`, per observation.
pub(super) fn scaled_groundings(
    columns: &PairColumns<'_, Option<&[f64]>, Option<&[f64]>>,
    // Per level `[rows | cols]` of the pair: the tested component and its factor.
    levels: &[Option<(usize, f64)>],
    n_rows: usize,
    n_components: usize,
) -> Vec<Grounding> {
    let mut residual = vec![0.0; n_components];
    let mut diagonal = vec![0.0; n_components];
    for o in columns.observations().filter(|o| o.w != 0.0) {
        match (levels[o.row], levels[n_rows + o.col]) {
            (Some((k, fi)), Some((l, fj))) if k == l => {
                let (p, q) = (fi * o.x, fj * o.y);
                residual[k] += o.w * (p + q) * (p + q);
                diagonal[k] += o.w * (p * p + q * q);
            }
            (row, col) => {
                for (k, p) in [
                    row.map(|(k, f)| (k, f * o.x)),
                    col.map(|(k, f)| (k, f * o.y)),
                ]
                .into_iter()
                .flatten()
                {
                    residual[k] += o.w * p * p;
                    diagonal[k] += o.w * p * p;
                }
            }
        }
    }
    residual
        .iter()
        .zip(&diagonal)
        .map(|(&r, &d)| {
            // An untested component reads `d = 0`: it never floats.
            if d > 0.0 && d.is_finite() && r <= SCALED_FLOATING_BOUND * d {
                Grounding::Floating
            } else {
                Grounding::Grounded
            }
        })
        .collect()
}
