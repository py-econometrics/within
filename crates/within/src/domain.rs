//! Domain layer: [`Design`] (design-matrix metadata) and factor-pair [`Subdomain`] construction.

pub(crate) mod cross_tab;
mod effect;
pub(crate) mod factor_pairs;

pub(crate) use cross_tab::{find_all_active_levels, BlockDiagonals, CrossTab};

pub use effect::Effect;

pub(crate) use factor_pairs::{
    build_local_domains, CoordinateMap, GroundEdges, Grounding, LocalComponent, LocalDomain,
    Reduction, SolveSpace,
};

// ===========================================================================
// Design — categorical fixed-effects design (data + layout)
// ===========================================================================

use std::borrow::Cow;

use crate::observation::ObservationFrame;
use crate::BuildError;

/// Per-term metadata: level count, coefficient-block offset, and column
/// structure. Columns are ordered `[intercept?, slopes…]`; coefficient `c` of
/// level `level` lives at `offset + c * n_levels + level`.
#[derive(Debug, Clone)]
pub(crate) struct TermMeta {
    pub n_levels: usize,
    pub offset: usize,
    /// Non-decreasing in the design's internal row order (fixed at construction).
    pub sorted: bool,
    pub intercept: bool,
    /// This term's slope loadings, as indices into the frame's continuous columns.
    pub slopes: Vec<usize>,
}

impl TermMeta {
    /// Coefficient-column count (`intercept? + slopes`), ordered
    /// `[intercept?, slopes…]`.
    pub fn n_columns(&self) -> usize {
        usize::from(self.intercept) + self.slopes.len()
    }

    pub fn n_dofs(&self) -> usize {
        self.n_columns() * self.n_levels
    }

    /// Global DOF base of coefficient column `column`.
    pub fn column_base(&self, column: usize) -> usize {
        self.offset + column * self.n_levels
    }
}

/// One coefficient column of a term: the intercept (`loading: None`, loading
/// value 1) or one slope column (`loading` = frame continuous column index).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Channel {
    pub term: usize,
    pub column: usize,
    pub loading: Option<usize>,
}

/// A cross-factor channel pair: one Gramian cross-block per pair.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChannelPair {
    pub q: Channel,
    pub r: Channel,
}

/// Fixed-effects design: observation columns plus coefficient-space layout.
#[derive(Clone, Debug)]
pub struct Design<'a> {
    /// Columns in internal row order (caller's, or an owned locality-sorted copy).
    pub(crate) frame: ObservationFrame<'a>,
    pub(crate) terms: Vec<TermMeta>,
    pub(crate) n_obs: usize,
    pub(crate) n_dofs: usize,
    /// `obs_perm[k]` = caller's original index of the observation at internal position `k`.
    pub(crate) obs_perm: Option<Vec<u32>>,
}

impl<'a> Design<'a> {
    /// Lower effect terms into a design: level columns plus each term's slope
    /// loadings, laid out term-major (`offset[t] + c * L_t + level`).
    pub fn new(effects: impl IntoIterator<Item = Effect<'a>>) -> Result<Self, BuildError> {
        let mut categorical: Vec<Cow<'a, [u32]>> = Vec::new();
        let mut continuous: Vec<Cow<'a, [f64]>> = Vec::new();
        let mut columns: Vec<(bool, Vec<usize>)> = Vec::new();
        for effect in effects {
            let slots = (continuous.len()..continuous.len() + effect.slopes().len()).collect();
            continuous.extend(effect.slopes().iter().map(|&z| Cow::Borrowed(z)));
            columns.push((effect.intercept(), slots));
            categorical.push(Cow::Borrowed(effect.levels()));
        }
        let frame = ObservationFrame::new(categorical, continuous)?;
        Self::build(frame, columns, true)
    }

    /// Construct from a frame of plain factors (each an intercept-only term),
    /// inferring each factor's level count (`max + 1`); locality-sorts all
    /// columns when the dominant factor is unsorted.
    pub fn from_frame(frame: ObservationFrame<'a>) -> Result<Self, BuildError> {
        let columns = vec![(true, Vec::new()); frame.n_factors()];
        Self::build(frame, columns, true)
    }

    /// [`from_frame`](Self::from_frame) without the locality sort — profiling escape hatch.
    #[doc(hidden)]
    pub fn from_frame_unsorted(frame: ObservationFrame<'a>) -> Result<Self, BuildError> {
        let columns = vec![(true, Vec::new()); frame.n_factors()];
        Self::build(frame, columns, false)
    }

    /// `column_structure[q]` = the term's `(intercept, slope column indices)`,
    /// aligned with the frame's categorical columns.
    fn build(
        frame: ObservationFrame<'a>,
        column_structure: Vec<(bool, Vec<usize>)>,
        locality_sort: bool,
    ) -> Result<Self, BuildError> {
        if frame.n_obs() == 0 {
            return Err(BuildError::EmptyObservations);
        }
        debug_assert_eq!(column_structure.len(), frame.n_factors());

        let n_obs = frame.n_obs();
        let mut terms = Vec::with_capacity(frame.n_factors());
        let mut offset = 0;
        for (q, (intercept, slopes)) in column_structure.into_iter().enumerate() {
            let col = frame.level_column(q);
            let mut max = 0;
            let mut sorted = true;
            let mut prev = 0;
            for &v in col {
                max = max.max(v);
                sorted &= v >= prev;
                prev = v;
            }
            let meta = TermMeta {
                n_levels: max as usize + 1,
                offset,
                sorted,
                intercept,
                slopes,
            };
            offset += meta.n_dofs();
            terms.push(meta);
        }

        // Sort by the term contributing the most DOFs (for plain factors, the
        // highest-cardinality one) so its gather/scatter runs sequentially.
        // `obs_perm` indexes observations as u32; beyond u32::MAX rows skip
        // the optimization — the solve itself has no such limit.
        let dominant = (0..terms.len()).max_by_key(|&q| terms[q].n_dofs());
        let (frame, obs_perm) = match dominant {
            Some(d) if locality_sort && !terms[d].sorted && u32::try_from(n_obs).is_ok() => {
                // Stable argsort. Must be `sort_by_cached_key`, NOT `sort_by_key`:
                // the latter re-gathers `key[i]` O(n log n) times and dominated
                // setup at tens of millions of rows.
                let key = frame.level_column(d);
                let mut perm: Vec<u32> = (0..n_obs as u32).collect();
                perm.sort_by_cached_key(|&i| key[i as usize]);
                let sorted_frame = frame.permuted(&perm);
                // Rescan sortedness: factors nested in (or duplicating) the
                // dominant one come out sorted, keeping their coalesced scatter.
                for (q, meta) in terms.iter_mut().enumerate() {
                    meta.sorted = sorted_frame.level_column(q).is_sorted();
                }
                (sorted_frame, Some(perm))
            }
            _ => (frame, None),
        };

        Ok(Design {
            frame,
            terms,
            n_obs,
            n_dofs: offset,
            obs_perm,
        })
    }

    /// Convert the frame's columns to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> Design<'static> {
        Design {
            frame: self.frame.into_owned(),
            terms: self.terms,
            n_obs: self.n_obs,
            n_dofs: self.n_dofs,
            obs_perm: self.obs_perm,
        }
    }

    /// Validate that an optional weight slice matches this design's observation count.
    pub(crate) fn validate_weights(&self, weights: Option<&[f64]>) -> Result<(), BuildError> {
        if let Some(w) = weights {
            if w.len() != self.n_obs {
                return Err(BuildError::WeightCountMismatch {
                    expected: self.n_obs,
                    got: w.len(),
                });
            }
            // `W^{1/2}` is applied to the design, so each weight must be finite and
            // non-negative; otherwise `sqrt(w)` is NaN and the solution is silently
            // corrupted. `wi >= 0.0` already rejects NaN (comparisons with NaN are
            // false); `is_finite` additionally rejects `+∞`.
            if let Some((index, &value)) = w
                .iter()
                .enumerate()
                .find(|&(_, &wi)| !(wi >= 0.0 && wi.is_finite()))
            {
                return Err(BuildError::InvalidWeight { index, value });
            }
        }
        Ok(())
    }

    /// Caller order → internal order: `out[k] = v[obs_perm[k]]`; borrows when unpermuted.
    pub(crate) fn permute_obs_in<'v>(&self, v: &'v [f64]) -> Cow<'v, [f64]> {
        debug_assert_eq!(v.len(), self.n_obs);
        match &self.obs_perm {
            None => Cow::Borrowed(v),
            Some(perm) => Cow::Owned(perm.iter().map(|&i| v[i as usize]).collect()),
        }
    }

    /// Internal order → caller order: `out[obs_perm[k]] = v[k]`.
    pub(crate) fn permute_obs_out(&self, v: Vec<f64>) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.n_obs);
        match &self.obs_perm {
            None => v,
            Some(perm) => {
                let mut out = vec![0.0; v.len()];
                for (k, &orig) in perm.iter().enumerate() {
                    out[orig as usize] = v[k];
                }
                out
            }
        }
    }

    /// Number of categorical factors in the design.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.terms.len()
    }

    /// The term's coefficient columns in `[intercept?, slopes…]` order.
    pub(crate) fn channels(&self, term: usize) -> impl Iterator<Item = Channel> + '_ {
        let meta = &self.terms[term];
        let first_slope = usize::from(meta.intercept);
        (0..first_slope + meta.slopes.len()).map(move |column| Channel {
            term,
            column,
            loading: (column >= first_slope).then(|| meta.slopes[column - first_slope]),
        })
    }

    /// Number of observations (rows of D).
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Total degrees of freedom (columns of D).
    #[inline]
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::ObservationFrame;

    fn frame(categorical: Vec<Vec<u32>>, continuous: Vec<Vec<f64>>) -> ObservationFrame<'static> {
        ObservationFrame::new(
            categorical.into_iter().map(Into::into).collect(),
            continuous.into_iter().map(Into::into).collect(),
        )
        .unwrap()
    }

    #[test]
    fn validate_weights_checks_count_and_finiteness() {
        let design = Design::from_frame(frame(vec![vec![0, 0, 0, 0, 0]], vec![])).unwrap();
        assert!(design.validate_weights(None).is_ok());
        assert!(design
            .validate_weights(Some(&[1.0, 2.0, 3.0, 4.0, 5.0]))
            .is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(design
            .validate_weights(Some(&[0.0, 1.0, 2.0, 3.0, 4.0]))
            .is_ok());
        // Length mismatch.
        assert!(design.validate_weights(Some(&[1.0, 2.0])).is_err());
        // Negative / non-finite weights are rejected with the offending index.
        assert!(matches!(
            design.validate_weights(Some(&[1.0, -2.0, 3.0, 4.0, 5.0])),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            design.validate_weights(Some(&[1.0, 2.0, f64::NAN, 4.0, 5.0])),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            design.validate_weights(Some(&[1.0, 2.0, 3.0, f64::INFINITY, 5.0])),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }

    #[test]
    fn from_frame_sorts_owned_unsorted_dominant() {
        // Factor 0 (3 levels) dominates and is unsorted; factor 1 starts sorted.
        let design =
            Design::from_frame(frame(vec![vec![2, 0, 1, 0], vec![0, 0, 1, 1]], vec![])).unwrap();

        // Stable argsort of [2,0,1,0] → original indices [1,3,2,0].
        assert_eq!(design.obs_perm.as_deref(), Some(&[1u32, 3, 2, 0][..]));
        assert!(design.terms[0].sorted);
        // Factor 1's permuted column [0,1,1,0] is no longer non-decreasing.
        assert!(!design.terms[1].sorted);

        assert_eq!(design.frame.level_column(0), [0, 0, 1, 2]);
        assert_eq!(design.frame.level_column(1), [0, 1, 1, 0]);
    }

    #[test]
    fn rescan_marks_nested_factor_sorted_after_permutation() {
        // Factor 1 nested in dominant factor 0 (level = col0 / 2): sorting by
        // factor 0 also sorts factor 1; the rescan must detect that.
        let col0 = vec![3u32, 0, 2, 1];
        let col1: Vec<u32> = col0.iter().map(|&v| v / 2).collect();
        let design = Design::from_frame(frame(vec![col0, col1], vec![])).unwrap();
        assert!(design.obs_perm.is_some());
        assert!(design.terms[0].sorted);
        assert!(design.terms[1].sorted);
    }

    #[test]
    fn from_frame_keeps_sorted_input() {
        let design =
            Design::from_frame(frame(vec![vec![0, 0, 1, 2], vec![1, 0, 1, 0]], vec![])).unwrap();
        assert!(design.obs_perm.is_none());
        assert!(design.terms[0].sorted);
        assert!(!design.terms[1].sorted);
    }

    #[test]
    fn continuous_column_stays_row_aligned_after_locality_sort() {
        let design = Design::from_frame(frame(
            vec![vec![2, 0, 1, 0]],
            vec![vec![10.0, 20.0, 30.0, 40.0]],
        ))
        .unwrap();

        let perm = design.obs_perm.as_ref().expect("permutation applied");
        assert_eq!(perm, &[1, 3, 2, 0]);
        assert_eq!(design.frame.level_column(0), [0, 0, 1, 2]);
        assert_eq!(design.frame.loading_column(0), [20.0, 40.0, 30.0, 10.0]);
    }

    #[test]
    fn new_lays_out_slope_terms_term_major() {
        // Term 0 is dominant (most DOFs); sorted levels keep the locality
        // sort a no-op so the frame columns stay in caller order.
        let f0 = [0u32, 0, 1, 1];
        let f1 = [0u32, 2, 1, 0];
        let z0 = [1.0, 2.0, 3.0, 4.0];
        let z1 = [5.0, 6.0, 7.0, 8.0];
        let effects = vec![
            Effect::new(&f0, true, [&z0[..], &z1[..]]).unwrap(),
            Effect::new(&f1, true, []).unwrap(),
            Effect::new(&f0, false, [&z1[..]]).unwrap(),
        ];
        let design = Design::new(effects).unwrap();

        // term 0: [intercept, z0, z1] over 2 levels; term 1: intercept over 3;
        // term 2: slope-only over 2.
        assert_eq!(design.terms[0].offset, 0);
        assert_eq!(design.terms[0].n_dofs(), 6);
        assert_eq!(design.terms[1].offset, 6);
        assert_eq!(design.terms[1].n_dofs(), 3);
        assert_eq!(design.terms[2].offset, 9);
        assert!(!design.terms[2].intercept);
        assert_eq!(design.terms[2].n_dofs(), 2);
        assert_eq!(design.n_dofs, 11);

        // slope indices resolve to the effects' loading columns in the frame.
        assert_eq!(design.terms[0].slopes, vec![0, 1]);
        assert_eq!(design.terms[2].slopes, vec![2]);
        assert_eq!(design.frame.loading_column(0), &z0[..]);
        assert_eq!(design.frame.loading_column(2), &z1[..]);
    }
}
