mod buffers;
mod executor;
mod planning;
mod preconditioner;
#[cfg(feature = "serde")]
mod serde;

pub use planning::{AdditiveSchwarzDiagnostics, ReductionStrategy};
pub use preconditioner::SchwarzPreconditioner;
