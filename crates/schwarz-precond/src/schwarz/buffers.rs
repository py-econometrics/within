//! Free lists that let the additive Schwarz apply reuse its buffers instead of allocating.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Free buffers; keeps at most `max_free` for reuse.
pub(super) struct Pool<B> {
    max_free: usize,
    free: Mutex<Vec<B>>,
}

impl<B> Pool<B> {
    pub(super) fn new(max_free: usize) -> Self {
        Self {
            max_free,
            free: Mutex::default(),
        }
    }

    /// Lends `f` a free buffer, or one from `make`, and keeps it for the next caller if there is room.
    pub(super) fn lend<R>(&self, make: impl FnOnce() -> B, f: impl FnOnce(&mut B) -> R) -> R {
        // A separate statement releases the lock before a miss allocates.
        let free = self.free_list().pop();
        let mut buffer = free.unwrap_or_else(make);
        let result = f(&mut buffer);
        let mut free = self.free_list();
        if free.len() < self.max_free {
            free.push(buffer);
        }
        result
    }

    // A panic elsewhere cannot leave a list of whole buffers inconsistent.
    fn free_list(&self) -> MutexGuard<'_, Vec<B>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_at_most_max_free_buffers() {
        let pool = Pool::new(2);
        pool.lend(|| (), |_| pool.lend(|| (), |_| pool.lend(|| (), |_| ())));
        assert_eq!(pool.free_list().len(), 2);
    }
}
