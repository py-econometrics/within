use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::local_solve::{LocalSolver, SubdomainEntry};

use super::executor::AdditiveExecutor;
use super::planning::{AdditiveScheduler, ReductionStrategy};
use super::preconditioner::SchwarzPreconditioner;

impl<S: LocalSolver + Serialize> Serialize for SchwarzPreconditioner<S> {
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        let mut state = serializer.serialize_struct("SchwarzPreconditioner", 3)?;
        state.serialize_field("subdomains", &*self.executor.subdomains)?;
        state.serialize_field("n_dofs", &self.executor.n_dofs)?;
        state.serialize_field("max_scratch_size", &self.executor.max_scratch_size)?;
        state.end()
    }
}

impl<'de, S: LocalSolver + serde::de::DeserializeOwned> Deserialize<'de>
    for SchwarzPreconditioner<S>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = "S: serde::de::DeserializeOwned"))]
        struct Helper<S: LocalSolver> {
            subdomains: Vec<SubdomainEntry<S>>,
            n_dofs: usize,
            max_scratch_size: usize,
        }

        let h: Helper<S> = Helper::deserialize(deserializer)?;
        Ok(SchwarzPreconditioner {
            reduction_strategy: ReductionStrategy::default(),
            scheduler: AdditiveScheduler::from_entries(&h.subdomains, h.n_dofs),
            executor: AdditiveExecutor::new(h.subdomains, h.n_dofs, h.max_scratch_size),
        })
    }
}
