//! Channel-pair subdomain construction.
//!
//! Each cross-factor channel pair becomes a Schwarz subdomain (one per
//! connected component of its bipartite cross-tab). Overlap is handled by
//! partition-of-unity weights — see [`schwarz_precond::domain`] for the math.
//!
//! Entry point: [`build_local_domains`].

use schwarz_precond::{PartitionWeights, SubdomainCore};

use crate::BuildError;

use super::{find_all_active_levels, BlockDiagonals, Channel, ChannelPair, CrossTab, Design};

mod routing;
use routing::{balance_and_scale, Frustrated};

/// The folded operator's null space — a computed property of the component, not
/// a chosen policy — fixing the local solve's gauge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// Zero-surplus (singular) Laplacian: null space is the constant, so the
    /// solve projects onto its complement (symmetric mean subtraction). Every
    /// plain component lands here; so do balanced singular signed ones.
    #[default]
    Constant,
    /// Diagonal surplus makes the operator nonsingular (null space `{0}`): the
    /// reduced solve is grounded exactly, no outer projection.
    Trivial,
}

/// Per-component congruence and kernel consumed by the local solve.
///
/// The congruence `d = σ ⊙ λ` (balancing signature times diagonal scaling, over
/// the `[q | r]` local DOF layout) turns a signed component's Gram into the
/// plain sign convention: the factorization is built on `d·A·d` and each solve
/// is sandwiched with `d`, so the local operator realizes the
/// congruence-transformed pseudo-solve `D Â⁺ D`.
#[derive(Clone, Debug, Default)]
pub(crate) struct ComponentTransform {
    /// `d = σ ⊙ λ` per local DOF; `None` is the identity (plain pairs).
    pub congruence: Option<Box<[f64]>>,
    /// Null space of the folded operator, fixing the solve's gauge.
    pub kernel: Kernel,
}

/// A local subdomain corresponding to a pair of factors.
#[derive(Clone)]
pub(crate) struct Subdomain {
    /// Generic subdomain core: global DOF indices, restriction, and partition-of-unity weights.
    pub core: SubdomainCore,
    /// Per-component congruence + kernel for the local solve.
    pub transform: ComponentTransform,
}

impl std::fmt::Debug for Subdomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subdomain")
            .field("n_dofs", &self.core.n_local())
            .finish()
    }
}

/// A factor-pair subdomain paired with the CrossTab it was built from.
///
/// The CrossTab (and its build-time [`BlockDiagonals`]) travel with the
/// subdomain so the local solver can be built without rebuilding the
/// cross-tabulation.
#[derive(Clone)]
pub(crate) struct LocalDomain {
    pub(crate) subdomain: Subdomain,
    pub(crate) cross_tab: CrossTab,
    pub(crate) block_diagonals: BlockDiagonals,
}

/// Build local subdomains (with pre-built CrossTabs) for cross-factor channel
/// pairs.
///
/// For each pair of distinct terms `q < r`, every channel of `q` is paired
/// with every channel of `r` (same-factor channel pairs are exactly
/// orthogonal after whitening, so they are never enumerated). Each channel
/// pair builds a fused CrossTab via one observation scan, detects connected
/// components on the bipartite structure, and creates one subdomain per
/// component. The CrossTab travels with each subdomain to avoid a rebuild.
///
/// Channel pairs are processed in parallel via Rayon. The
/// `compute_partition_weights` step remains sequential after the parallel
/// collect.
pub(crate) fn build_local_domains(
    design: &Design<'_>,
    weights: Option<&[f64]>,
) -> Result<Vec<LocalDomain>, BuildError> {
    use rayon::prelude::*;

    let channels: Vec<Channel> = (0..design.n_factors())
        .flat_map(|term| design.channels(term))
        .collect();
    let pairs: Vec<ChannelPair> = channels
        .iter()
        .enumerate()
        .flat_map(|(i, &q)| {
            channels[i + 1..]
                .iter()
                .filter(move |r| r.term != q.term)
                .map(move |&r| ChannelPair { q, r })
        })
        .collect();
    let all_active = find_all_active_levels(design);

    let per_pair: Vec<Vec<LocalDomain>> = pairs
        .par_iter()
        .map(|&pair| {
            let Some((full_ct, full_diag, l2g)) =
                CrossTab::build_for_pair_with_active(design, weights, pair, &all_active)
            else {
                return Ok(Vec::new());
            };
            split_into_subdomains(pair, full_ct, full_diag, &l2g)
        })
        .collect::<Result<_, BuildError>>()?;
    let mut domain_pairs: Vec<LocalDomain> = per_pair.into_iter().flatten().collect();

    compute_partition_weights(&mut domain_pairs, design.n_dofs);

    Ok(domain_pairs)
}

/// Split a full CrossTab into per-component subdomains.
///
/// Finds bipartite connected components, extracts a sub-CrossTab and its
/// sliced [`BlockDiagonals`] for each, and routes every component through
/// [`balance_and_scale`]; frustration is a build error. Plain pairs shortcut
/// to the identity transform routing would compute anyway (all cells
/// non-negative and exactly row-singular), skipping its passes. Dead
/// singletons (zero diagonal — an exact-zero design column) produce no
/// subdomain, matching the uncovered-inactive-level invariant.
fn split_into_subdomains(
    pair: ChannelPair,
    full_ct: CrossTab,
    full_diag: BlockDiagonals,
    l2g: &[u32],
) -> Result<Vec<LocalDomain>, BuildError> {
    let n_q_full = full_ct.n_q();
    let components = full_ct.bipartite_connected_components();

    let (cross_tabs, diagonals): (Vec<CrossTab>, Vec<BlockDiagonals>) = if components.len() == 1 {
        (vec![full_ct], vec![full_diag])
    } else {
        // One reusable remap buffer pair for the whole parent; `extract_component`
        // resets it per component, avoiding a fresh parent-sized allocation each.
        let mut q_remap = vec![u32::MAX; full_ct.n_q()];
        let mut r_remap = vec![u32::MAX; full_ct.n_r()];
        let cross_tabs = components
            .iter()
            .map(|comp| full_ct.extract_component(comp, &mut q_remap, &mut r_remap))
            .collect();
        let diagonals = components
            .iter()
            .map(|comp| full_diag.extract_component(comp))
            .collect();
        (cross_tabs, diagonals)
    };

    let mut domains = Vec::with_capacity(components.len());
    for ((comp, comp_ct), comp_diag) in components.iter().zip(cross_tabs).zip(diagonals) {
        if comp_diag.q.iter().chain(&comp_diag.r).all(|&v| v == 0.0) {
            continue;
        }
        let transform = if pair.is_plain() {
            ComponentTransform::default()
        } else {
            balance_and_scale(&comp_ct, &comp_diag).map_err(|Frustrated| {
                BuildError::FrustratedComponent {
                    term_q: pair.q.term,
                    column_q: pair.q.column,
                    term_r: pair.r.term,
                    column_r: pair.r.column,
                }
            })?
        };
        let comp_l2g: Vec<u32> = comp
            .q_indices
            .iter()
            .map(|&i| l2g[i])
            .chain(comp.r_indices.iter().map(|&i| l2g[n_q_full + i]))
            .collect();
        domains.push(LocalDomain {
            subdomain: Subdomain {
                core: schwarz_precond::SubdomainCore::uniform(comp_l2g),
                transform,
            },
            cross_tab: comp_ct,
            block_diagonals: comp_diag,
        });
    }
    Ok(domains)
}

/// Compute partition-of-unity weights for overlapping Schwarz subdomains.
///
/// The two-sided additive Schwarz formula `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`
/// requires that the squared weights sum to identity at every DOF:
/// `Σ Rᵢᵀ D̃ᵢ² Rᵢ = I`. For a DOF appearing in `c` subdomains, each weight
/// is set to `1/√c`, so that `c × (1/√c)² = 1`.
///
/// In the common (non-overlapping) case where every DOF belongs to exactly one
/// subdomain, all weights are 1.0 and the compact `PartitionWeights::Uniform`
/// representation is used to avoid per-DOF storage.
fn compute_partition_weights(domain_pairs: &mut [LocalDomain], n_dofs: usize) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Pass 1: histogram how many subdomains each DOF appears in. Atomic
    // increments commute, so the parallel accumulation matches the serial scan.
    let counts: Vec<AtomicU32> = (0..n_dofs).map(|_| AtomicU32::new(0)).collect();
    domain_pairs.par_iter().for_each(|ld| {
        for &idx in ld.subdomain.core.global_indices() {
            debug_assert!((idx as usize) < n_dofs);
            counts[idx as usize].fetch_add(1, Ordering::Relaxed);
        }
    });
    let counts: Vec<u32> = counts.into_iter().map(AtomicU32::into_inner).collect();

    // Pass 2: each subdomain's weights depend only on the shared counts, so the
    // per-domain work is independent.
    domain_pairs.par_iter_mut().for_each(|ld| {
        let d = &mut ld.subdomain;
        let all_unique = d
            .core
            .global_indices()
            .iter()
            .all(|&idx| counts[idx as usize] <= 1);
        if all_unique {
            d.core.set_uniform_partition_weights();
        } else {
            let weights: Vec<f64> = d
                .core
                .global_indices()
                .iter()
                .map(|&idx| {
                    let c = counts[idx as usize];
                    debug_assert!(c > 0);
                    1.0 / (c as f64).sqrt()
                })
                .collect();
            d.core
                .set_partition_weights(PartitionWeights::NonUniform(weights))
                .expect("partition weight count must match index count");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Design;
    use crate::observation::ObservationFrame;

    fn make_test_design() -> Design<'static> {
        let frame = ObservationFrame::new(
            vec![
                vec![0u32, 1, 2, 0, 1, 2].into(),
                vec![0u32, 1, 0, 1, 0, 1].into(),
                vec![0u32, 0, 1, 1, 0, 1].into(),
            ],
            Vec::new(),
        )
        .expect("valid frame");
        Design::from_frame(frame).expect("valid test design")
    }

    #[test]
    fn test_full_cover_domain_count() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None).expect("plain domains build");
        // 3 factor pairs; each pair may produce multiple components
        assert!(domain_pairs.len() >= 3);
    }

    #[test]
    fn test_partition_of_unity() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None).expect("plain domains build");
        let n_dofs = dm.n_dofs;
        // Two-sided PoU: squared weights must sum to 1 at every DOF.
        let mut weight_sq_sum = vec![0.0; n_dofs];
        for ld in &domain_pairs {
            let d = &ld.subdomain;
            for (i, &idx) in d.core.global_indices().iter().enumerate() {
                let w = d.core.partition_weights().get(i);
                weight_sq_sum[idx as usize] += w * w;
            }
        }
        for &ws in &weight_sq_sum {
            if ws > 0.0 {
                assert!((ws - 1.0).abs() < 1e-12, "Weight² sum {ws} != 1.0");
            }
        }
    }

    #[test]
    fn test_domains_cover_all_dofs() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None).expect("plain domains build");
        let mut covered = vec![false; dm.n_dofs];
        for ld in &domain_pairs {
            for &idx in ld.subdomain.core.global_indices() {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }
}
