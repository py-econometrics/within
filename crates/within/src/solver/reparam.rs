//! Within-level reparametrization of a design's sole varying-slope term.

use crate::domain::Design;

use super::UnidentifiedDirection;

/// Per-level affine change of basis for the design's sole slope column: the
/// solve sees `(z - center) / scale`, which zeroes each level's
/// intercept–slope cross-term and normalizes the slope block, leaving fitted
/// values invariant. [`Self::back_transform`] returns coefficients to the
/// user's parametrization.
pub(crate) struct SlopeReparam {
    offset: usize,
    n_levels: usize,
    intercept: bool,
    /// Per-level `(center, scale)`; `None` at dropped and empty levels, whose
    /// whitened column entries are zeroed and whose coefficients stay `0`.
    transforms: Vec<Option<(f64, f64)>>,
    /// Directions the data cannot identify, ascending in `(level, column)`.
    pub(crate) unidentified: Vec<UnidentifiedDirection>,
}

impl SlopeReparam {
    /// Whiten the sole slope term's loading column in place and record how to
    /// undo it. Returns `None` for slope-free designs.
    ///
    /// Identification is judged structurally over positive-weight rows so the
    /// dropped set is deterministic: an empty level drops the term's every
    /// column; a slope with no within-level variation (no nonzero value, for
    /// a slope-only term) drops its slope column. Dropped slope entries are
    /// materialized as exact zeros, so the minimal-norm LSMR solution leaves
    /// an exact `0` in the coefficient slot.
    pub(crate) fn build(design: &mut Design<'_>, weights: Option<&[f64]>) -> Option<Self> {
        let term = design.terms.iter().position(|t| !t.slopes.is_empty())?;
        let meta = &design.terms[term];
        debug_assert_eq!(meta.slopes.len(), 1, "reparametrization is V=1-only");
        let (offset, n_levels, intercept) = (meta.offset, meta.n_levels, meta.intercept);
        let z_col = meta.slopes[0];
        let levels = design.frame.level_column(term);
        let z = design.frame.loading_column(z_col);

        let mut stats = vec![LevelStats::default(); n_levels];
        for (i, (&level, &zi)) in levels.iter().zip(z).enumerate() {
            stats[level as usize].observe(zi, weights.map_or(1.0, |ws| ws[i]));
        }

        let slope_column = usize::from(intercept);
        let mut transforms = Vec::with_capacity(n_levels);
        let mut unidentified = Vec::new();
        for (level, s) in stats.iter().enumerate() {
            let transform = s.whitening(intercept);
            if transform.is_none() {
                // An empty level loses every column, a degenerate slope only its own.
                let first = if s.is_empty() { 0 } else { slope_column };
                for column in first..=slope_column {
                    unidentified.push(UnidentifiedDirection {
                        term,
                        level,
                        column,
                    });
                }
            }
            transforms.push(transform);
        }

        let whitened: Vec<f64> = levels
            .iter()
            .zip(z)
            .map(|(&level, &zi)| match transforms[level as usize] {
                Some((center, scale)) => (zi - center) / scale,
                None => 0.0,
            })
            .collect();
        design.frame.set_loading_column(z_col, whitened);

        Some(Self {
            offset,
            n_levels,
            intercept,
            transforms,
            unidentified,
        })
    }

    /// Map solve-basis coefficients back to the user's parametrization.
    pub(crate) fn back_transform(&self, x: &mut [f64]) {
        let slope_base = self.offset + if self.intercept { self.n_levels } else { 0 };
        for (l, t) in self.transforms.iter().enumerate() {
            // Dropped levels were never transformed; their coefficients are 0.
            let Some((center, scale)) = *t else { continue };
            let b = x[slope_base + l] / scale;
            x[slope_base + l] = b;
            if self.intercept {
                x[self.offset + l] -= b * center;
            }
        }
    }
}

/// Weighted running moments of `z` within one level (Welford), plus the
/// structural variation flags, over positive-weight rows only.
#[derive(Clone, Copy, Default)]
struct LevelStats {
    w_sum: f64,
    mean: f64,
    m2: f64,
    first_z: f64,
    varies: bool,
}

impl LevelStats {
    fn observe(&mut self, z: f64, w: f64) {
        // Zero-weight rows may carry arbitrary values; they must not affect
        // identification or the scale.
        if w <= 0.0 {
            return;
        }
        if self.w_sum == 0.0 {
            self.first_z = z;
        } else {
            self.varies |= z != self.first_z;
        }
        self.w_sum += w;
        let delta = z - self.mean;
        self.mean += (w / self.w_sum) * delta;
        self.m2 += w * delta * (z - self.mean);
    }

    fn is_empty(&self) -> bool {
        self.w_sum == 0.0
    }

    /// The level's `(center, scale)`, or `None` when the slope direction is
    /// structurally unidentifiable: `z` constant against a pinned intercept,
    /// identically zero without one, or an empty level.
    fn whitening(&self, intercept: bool) -> Option<(f64, f64)> {
        let (identified, center, ssq) = if intercept {
            (self.varies, self.mean, self.m2)
        } else {
            // Uncentered second moment: Σwz² = m2 + w·mean², both non-negative.
            let ssq = self.m2 + self.w_sum * self.mean * self.mean;
            (self.varies || self.first_z != 0.0, 0.0, ssq)
        };
        let scale = ssq.sqrt();
        (identified && scale > 0.0 && scale.is_finite()).then_some((center, scale))
    }
}
