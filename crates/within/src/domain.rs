pub(crate) mod factor_pairs;
mod schema;

pub(crate) use factor_pairs::build_local_domains;
pub(crate) use factor_pairs::{build_domains_and_gramian_blocks, PairBlockData};
pub use schema::FixedEffectsDesign;
pub use schema::WeightedDesign;

// Re-exports from schwarz-precond
pub use schwarz_precond::PartitionWeights;
pub use schwarz_precond::SubdomainCore;

/// A local subdomain corresponding to a pair of factors.
#[derive(Clone)]
pub struct Subdomain {
    pub factor_pair: (usize, usize),
    pub core: SubdomainCore,
}

impl std::fmt::Debug for Subdomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subdomain")
            .field("factor_pair", &self.factor_pair)
            .field("n_dofs", &self.core.global_indices.len())
            .finish()
    }
}
