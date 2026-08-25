//! Gauge directions behind a cross-term collinearity warning (#297): a covariate both terms
//! reproduce per level gives `D v = 0` for `v = (fit on its own term) − (fit on the other)`.

use super::Design;
use crate::BuildWarning;

/// One warned pair's covariate, centered so the two fits differ by the aliasing alone
/// and weighted so the design's adjoint reads it directly.
pub(crate) struct GaugeCandidate {
    /// Index of the warning this came from, which names both terms and carries the verdict back.
    pub(crate) warning: usize,
    pub(crate) weighted: Vec<f64>,
}

/// Must run before whitening, which overwrites the covariate's column.
pub(crate) fn gauge_candidates(
    design: &Design<'_>,
    weights: Option<&[f64]>,
    warnings: &[BuildWarning],
) -> Vec<GaugeCandidate> {
    warnings
        .iter()
        .enumerate()
        .filter_map(|(warning, w)| {
            let (slope, term) = match w {
                BuildWarning::CollinearSlopeCovariate { slope, term, .. } => (*slope, *term),
                _ => return None,
            };
            let covariate = *design.loading(slope).covariate()?;
            let c = design.loading_column(covariate as usize);
            let (mut sum, mut total) = (0.0, 0.0);
            for (obs, &ci) in c.iter().enumerate() {
                let w = weights.map_or(1.0, |ws| ws[obs]);
                sum += w * ci;
                total += w;
            }
            // Two intercepts alias through the ordinary FE gauge, not the covariate.
            let intercepts = [slope.term, term]
                .iter()
                .all(|&t| design.terms[t].intercept_column().is_some());
            let origin = match intercepts && total > 0.0 {
                true => sum / total,
                false => 0.0,
            };
            Some(GaugeCandidate {
                warning,
                weighted: c
                    .iter()
                    .enumerate()
                    .map(|(obs, &ci)| (ci - origin) * weights.map_or(1.0, |ws| ws[obs].sqrt()))
                    .collect(),
            })
        })
        .collect()
}
