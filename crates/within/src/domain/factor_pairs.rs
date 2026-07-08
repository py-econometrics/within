//! Channel-pair subdomain construction.
//!
//! Each cross-factor channel pair becomes a Schwarz subdomain (one per
//! connected component of its bipartite cross-tab). Overlap is handled by
//! partition-of-unity weights — see [`schwarz_precond::domain`] for the math.
//!
//! Entry point: [`build_local_domains`].

use schwarz_precond::{PartitionWeights, SubdomainCore};

use crate::BuildError;

use super::{find_all_active_levels, BlockDiagonals, ChannelPair, CrossTab, Design};

mod routing;
use routing::{balance_and_scale, Frustrated};

/// Kernel policy the local solve honors for one component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum KernelPolicy {
    /// Constant-kernel (plain) component: symmetric mean projection around the solve.
    #[default]
    MeanProjection,
    /// No projection (signed component): the reduced solve is grounded exactly.
    None,
}

/// Per-component congruence and kernel policy consumed by the local solve.
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
    /// Projection policy around the local solve.
    pub kernel: KernelPolicy,
}

/// A local subdomain corresponding to a pair of factors.
#[derive(Clone)]
pub(crate) struct Subdomain {
    /// Generic subdomain core: global DOF indices, restriction, and partition-of-unity weights.
    pub core: SubdomainCore,
    /// Per-component congruence + kernel policy for the local solve.
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

    let n_factors = design.n_factors();
    let pairs: Vec<ChannelPair> = (0..n_factors)
        .flat_map(|q| ((q + 1)..n_factors).map(move |r| (q, r)))
        .flat_map(|(q, r)| {
            design.channels(q).flat_map(move |cq| {
                design
                    .channels(r)
                    .map(move |cr| ChannelPair { q: cq, r: cr })
            })
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
            let n_q_full = full_ct.n_q();
            split_into_subdomains(pair, full_ct, full_diag, &l2g, n_q_full)
        })
        .collect::<Result<_, BuildError>>()?;
    let mut domain_pairs: Vec<LocalDomain> = per_pair.into_iter().flatten().collect();

    compute_partition_weights(&mut domain_pairs, design.n_dofs);

    Ok(domain_pairs)
}

/// Split a full CrossTab into per-component subdomains.
///
/// Finds bipartite connected components, extracts a sub-CrossTab and its sliced
/// [`BlockDiagonals`] for each, and builds a `Subdomain` with uniform
/// partition-of-unity weights and the per-component routing policy:
///
/// - dead singleton (edgeless, zero diagonal — an exact-zero design column):
///   skipped, matching the uncovered-inactive-level invariant;
/// - live singleton (positive diagonal, cross row cancelled — signed pairs
///   only): kept as a trivial 1×1 so `M⁻¹` has no zero row;
/// - plain multi-node: today's default transform, arithmetic untouched;
/// - signed multi-node: exact-Schur kernel policy plus the
///   [`balance_and_scale`] congruence; frustration is a build error.
fn split_into_subdomains(
    pair: ChannelPair,
    full_ct: CrossTab,
    full_diag: BlockDiagonals,
    l2g: &[u32],
    n_q_full: usize,
) -> Result<Vec<LocalDomain>, BuildError> {
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

    components
        .iter()
        .zip(cross_tabs)
        .zip(diagonals)
        .filter_map(|((comp, comp_ct), comp_diag)| {
            let transform = if comp_ct.n_local() == 1 {
                let diag = comp_diag.q.first().or(comp_diag.r.first());
                if *diag.expect("singleton has one diagonal") == 0.0 {
                    return None;
                }
                ComponentTransform {
                    congruence: None,
                    kernel: KernelPolicy::None,
                }
            } else if pair.is_plain() {
                ComponentTransform::default()
            } else {
                match balance_and_scale(&comp_ct, &comp_diag) {
                    Ok(congruence) => ComponentTransform {
                        congruence,
                        kernel: KernelPolicy::None,
                    },
                    Err(Frustrated) => {
                        return Some(Err(BuildError::FrustratedComponent {
                            term_q: pair.q.term,
                            column_q: pair.q.column,
                            term_r: pair.r.term,
                            column_r: pair.r.column,
                        }))
                    }
                }
            };
            let comp_l2g: Vec<u32> = comp
                .q_indices
                .iter()
                .map(|&i| l2g[i])
                .chain(comp.r_indices.iter().map(|&i| l2g[n_q_full + i]))
                .collect();
            let core = schwarz_precond::SubdomainCore::uniform(comp_l2g);
            Some(Ok(LocalDomain {
                subdomain: Subdomain { core, transform },
                cross_tab: comp_ct,
                block_diagonals: comp_diag,
            }))
        })
        .collect()
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
    use crate::block_elim::BlockElimSolver;
    use crate::config::{ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig};
    use crate::domain::{Design, Effect};
    use crate::observation::ObservationFrame;
    use schwarz_precond::LocalSolver;

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

    /// `f[z] + g` with `z` pre-whitened per f-level (`Σu = 0`, `Σu² = 1`),
    /// mirroring what `SlopeReparam::build` hands the preconditioner. The
    /// 2-level `g` makes the signed pair structurally balanced (centering
    /// forces opposite-signed row cells).
    fn slope_plus_binary_design() -> Design<'static> {
        const A: f64 = std::f64::consts::FRAC_1_SQRT_2;
        const F: [u32; 6] = [0, 0, 1, 1, 2, 2];
        const G: [u32; 6] = [0, 1, 1, 0, 0, 1];
        const U: [f64; 6] = [-A, A, -A, A, -A, A];
        let effects = vec![
            Effect::new(&F, true, [&U[..]]).unwrap(),
            Effect::new(&G, true, []).unwrap(),
        ];
        Design::new(effects).expect("valid slope design")
    }

    #[test]
    fn routes_signed_pair_with_congruence_and_exact_kernel() {
        let design = slope_plus_binary_design();
        let domains =
            build_local_domains(&design, None).expect("structurally balanced design builds");
        // One single-component domain per channel pair: (f-int, g-int) plain
        // and (f-slope, g-int) signed.
        assert_eq!(domains.len(), 2);

        // DOF layout: f-int 0..3, f-slope 3..6, g-int 6..8.
        let plain = domains
            .iter()
            .find(|d| d.subdomain.transform.kernel == KernelPolicy::MeanProjection)
            .expect("plain pair domain");
        assert!(plain.subdomain.transform.congruence.is_none());
        assert!(plain
            .subdomain
            .core
            .global_indices()
            .iter()
            .all(|&i| i < 3 || (6..8).contains(&i)));

        let signed = domains
            .iter()
            .find(|d| d.subdomain.transform.kernel == KernelPolicy::None)
            .expect("signed pair domain");
        assert!(signed
            .subdomain
            .core
            .global_indices()
            .iter()
            .all(|&i| (3..8).contains(&i)));
        let d = signed
            .subdomain
            .transform
            .congruence
            .as_deref()
            .expect("mixed-sign cells need a congruence");
        assert_eq!(d.len(), 5);
        let ct = &signed.cross_tab;
        for i in 0..ct.n_q() {
            for idx in ct.c.indptr[i] as usize..ct.c.indptr[i + 1] as usize {
                let j = ct.c.indices[idx] as usize;
                let v = d[i] * d[ct.n_q() + j] * ct.c.data[idx];
                assert!(v >= 0.0, "cell ({i},{j}) folds to {v}");
            }
        }

        // No zero-diagonal singleton survived routing.
        for dom in &domains {
            assert!(dom
                .block_diagonals
                .q
                .iter()
                .chain(&dom.block_diagonals.r)
                .all(|&v| v > 0.0));
        }
    }

    #[test]
    fn signed_component_solve_realizes_produced_congruence() {
        let design = slope_plus_binary_design();
        let domains = build_local_domains(&design, None).expect("builds");
        let signed = domains
            .into_iter()
            .find(|d| d.subdomain.transform.kernel == KernelPolicy::None)
            .expect("signed pair domain");

        // Raw (pre-congruence) A = [D_q, C; Cᵀ, D_r] as the solve oracle;
        // `BlockElimSolver::build` consumes and congruence-scales its copy.
        let ct_raw = signed.cross_tab.clone();
        let diag_raw = signed.block_diagonals.clone();
        let n_q = ct_raw.n_q();
        let n = ct_raw.n_local();
        let mut a = vec![0.0; n * n];
        for (i, v) in diag_raw.q.iter().chain(&diag_raw.r).enumerate() {
            a[i * n + i] = *v;
        }
        for i in 0..n_q {
            for idx in ct_raw.c.indptr[i] as usize..ct_raw.c.indptr[i + 1] as usize {
                let j = n_q + ct_raw.c.indices[idx] as usize;
                a[i * n + j] = ct_raw.c.data[idx];
                a[j * n + i] = ct_raw.c.data[idx];
            }
        }

        let config = LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            approx_schur: Some(ApproxSchurConfig::default()),
            dense_threshold: 8,
        };
        let solver = BlockElimSolver::build(
            signed.cross_tab,
            signed.block_diagonals,
            signed.subdomain.transform,
            &config,
        )
        .expect("signed local build");

        // The component is singular, so D·Â⁺·D is exact only on range(A):
        // take r = A·y and check A·x = r (gauge drops out).
        let y = [0.3, -1.1, 0.7, 0.25, -0.6];
        let r: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * y[j]).sum())
            .collect();

        let mut rhs = vec![0.0; solver.scratch_size()];
        rhs[..n].copy_from_slice(&r);
        let mut sol = vec![0.0; solver.scratch_size()];
        solver
            .solve_local(&mut rhs, &mut sol, false)
            .expect("signed solve_local");

        for i in 0..n {
            let ax: f64 = (0..n).map(|j| a[i * n + j] * sol[j]).sum();
            assert!(
                (ax - r[i]).abs() < 1e-8,
                "row {i}: A·x = {ax}, expected {}",
                r[i]
            );
        }
    }
}
