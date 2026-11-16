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
    pub const ONE: Self = Self::Count(NonZeroUsize::MIN);

    /// The most threads it runs at a time; the calling thread alone asks no pool.
    pub(crate) fn tasks(self) -> usize {
        match self {
            Self::ONE => 1,
            Self::All => rayon::current_num_threads(),
            Self::Count(n) => n.get().min(rayon::current_num_threads()),
        }
    }

    /// Whether more than one thread may run: the calling thread alone makes no rayon call.
    pub(crate) fn forks(self) -> bool {
        self.tasks() > 1
    }
}
