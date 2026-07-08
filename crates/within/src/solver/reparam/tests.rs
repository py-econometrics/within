//! Seam tests for the per-level pivoted Gram–Schmidt: pivot bookkeeping through
//! non-trivial pivot orders and the rank-tolerance contract, neither
//! reachable through the public API (the tolerance is a crate-internal
//! constant).

use super::*;

#[test]
fn gram_schmidt_orthonormalizes_under_a_non_monotonic_pivot_order() {
    // Diagonals [2, 5, 3] force the pivot sequence 1 → 2 → 0: any
    // bookkeeping that assumes natural or physically swapped order breaks
    // the W·G·Wᵀ = I identity below (the prototype's ≥3-slope cliff).
    let g = [2.0, 1.0, 0.5, 1.0, 5.0, 2.0, 0.5, 2.0, 3.0];
    let (w, kept) = pivoted_gram_schmidt(&g, 3, RANK_TOL);
    assert_eq!(kept, [true; 3]);
    assert_eq!(w.len(), 9);

    for r in 0..3 {
        for s in 0..3 {
            let wgw: f64 = (0..3)
                .flat_map(|j| (0..3).map(move |k| (j, k)))
                .map(|(j, k)| w[r * 3 + j] * g[j * 3 + k] * w[s * 3 + k])
                .sum();
            let expected = if r == s { 1.0 } else { 0.0 };
            assert!(
                (wgw - expected).abs() < 1e-12,
                "(W·G·Wᵀ)[{r}][{s}] = {wgw}, expected {expected}"
            );
        }
    }
}

#[test]
fn zero_tolerance_keeps_a_near_degenerate_direction_the_default_drops() {
    let eps = 1e-12;
    let g = [1.0, 1.0 - eps, 1.0 - eps, 1.0];
    let (_, kept) = pivoted_gram_schmidt(&g, 2, RANK_TOL);
    assert_eq!(kept.iter().filter(|&&k| k).count(), 1);
    let (_, kept) = pivoted_gram_schmidt(&g, 2, 0.0);
    assert_eq!(kept, [true, true]);
}
