//! How many threads a local solve may use.

use std::num::NonZeroUsize;

/// The threads a local solve may use; no result's bits depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    /// Every thread of the caller's rayon pool.
    All,
    /// At most `n` threads at a time; `Count(1)` runs on the calling thread.
    Count(NonZeroUsize),
}

impl Threads {
    /// The calling thread alone.
    pub(crate) const ONE: Self = Self::Count(NonZeroUsize::MIN);
}
