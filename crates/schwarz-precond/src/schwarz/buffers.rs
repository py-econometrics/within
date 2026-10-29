//! Free lists that let the additive Schwarz apply reuse its buffers instead of allocating.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Free buffers of one length; keeps at most `max_free` for reuse.
pub(super) struct Pool {
    len: usize,
    max_free: usize,
    free: Mutex<Vec<Vec<f64>>>,
}

impl Pool {
    pub(super) fn new(len: usize, max_free: usize) -> Self {
        Self {
            len,
            max_free,
            free: Mutex::default(),
        }
    }

    /// Lends `f` a free buffer, or a new one, and keeps it for the next caller if there is room.
    pub(super) fn lend<R>(&self, f: impl FnOnce(&mut [f64]) -> R) -> R {
        // A separate statement releases the lock before a miss allocates.
        let free = self.free_list().pop();
        let mut buffer = free.unwrap_or_else(|| vec![0.0; self.len]);
        let result = f(&mut buffer);
        let mut free = self.free_list();
        if free.len() < self.max_free {
            free.push(buffer);
        }
        result
    }

    // A panic elsewhere cannot leave a list of whole buffers inconsistent.
    fn free_list(&self) -> MutexGuard<'_, Vec<Vec<f64>>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_at_most_max_free_buffers() {
        let pool = Pool::new(1, 2);
        pool.lend(|_| pool.lend(|_| pool.lend(|_| ())));
        assert_eq!(pool.free_list().len(), 2);
    }
}
