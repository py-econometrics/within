use std::time::Instant;

use approx_chol::Side;
use rayon::prelude::*;
use schwarz_precond::SubdomainCore;

use super::{build_additive, FeSchwarz, SchwarzConfig};
use crate::channel::ChannelPair;
use crate::domain::elimination::Elimination;
use crate::domain::factor_pairs::{
    channel_pairs, compute_partition_weights, pair_domains, PairMatrix,
};
use crate::domain::LocalDomain;
use crate::{BuildError, BuildWarning};

/// Which factor pairs the reduced inverse keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pairs {
    /// Every pair: Schur complements plus the retained pairs' Grams, partition-weighted.
    All,
    /// Only the pairs split by the eliminated factor: block Jacobi on `S`.
    SchurOnly,
}

/// The additive inverse on the kept coefficients and what its build reported.
pub(crate) struct PairSchur {
    pub(crate) map: FeSchwarz,
    pub(crate) warnings: Vec<BuildWarning>,
    pub(crate) seconds: f64,
    pub(crate) schur_domains: usize,
    pub(crate) retained_domains: usize,
}

/// The eliminated factor's pairs factor their complement onto the other side; the rest their Gram, unless dropped.
fn pair_matrix(e: &Elimination<'_>, pair: ChannelPair, pairs: Pairs) -> Option<PairMatrix> {
    match (pair.rows.term == e.term(), pair.cols.term == e.term()) {
        (true, _) => Some(PairMatrix::Complement(Side::Right)),
        (_, true) => Some(PairMatrix::Complement(Side::Left)),
        _ if pairs == Pairs::SchurOnly => None,
        _ => Some(PairMatrix::Gram),
    }
}

/// `domain` over the kept coefficients: no complement or retained Gram touches the eliminated ones.
fn reduced(domain: LocalDomain, e: &Elimination<'_>) -> LocalDomain {
    let removed = e.removed();
    let (shift, end) = (removed.len() as u32, removed.end as u32);
    let indices = (domain.core.global_indices().iter())
        .map(|&i| {
            debug_assert!(!removed.contains(&(i as usize)), "a kept coefficient");
            if i >= end {
                i - shift
            } else {
                i
            }
        })
        .collect();
    LocalDomain {
        core: SubdomainCore::uniform(indices),
        solver: domain.solver,
    }
}

pub(crate) fn build_pair_schur(
    e: &Elimination<'_>,
    pairs: Pairs,
    config: &SchwarzConfig,
) -> Result<PairSchur, BuildError> {
    let start = Instant::now();
    let approx_chol = config.local_solver.to_approx_chol()?;
    let (_, all_pairs) = channel_pairs(e.prepared());
    let per_pair = (all_pairs.par_iter())
        .filter_map(|&pair| {
            let matrix = pair_matrix(e, pair, pairs)?;
            let built = pair_domains(
                e.prepared(),
                pair,
                approx_chol,
                &config.local_solver,
                matrix,
            );
            Some(built.map(|(domains, warnings)| (matrix, domains, warnings)))
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
    let (mut domains, mut warnings) = (Vec::new(), Vec::new());
    let mut schur_domains = 0;
    for (matrix, pair_domains, pair_warnings) in per_pair {
        if let PairMatrix::Complement(_) = matrix {
            schur_domains += pair_domains.len();
        }
        domains.extend(pair_domains.into_iter().map(|d| reduced(d, e)));
        warnings.extend(pair_warnings);
    }
    let retained_domains = domains.len() - schur_domains;
    compute_partition_weights(&mut domains, e.n_kept());
    Ok(PairSchur {
        map: build_additive(domains, config, e.n_kept())?,
        warnings,
        seconds: start.elapsed().as_secs_f64(),
        schur_domains,
        retained_domains,
    })
}
