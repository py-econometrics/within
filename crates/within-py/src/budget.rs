//! A call's thread budget: the rayon pool it runs on.

use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyResult;
use rayon::ThreadPool;

/// One pool per thread count, built on a call's first use and kept for the process.
static POOLS: Mutex<BTreeMap<usize, &'static ThreadPool>> = Mutex::new(BTreeMap::new());

/// The pool a call runs on: the pool it is called from, or the cached pool of the caller's count.
#[derive(Clone, Copy)]
pub(crate) struct ThreadBudget(Option<&'static ThreadPool>);

impl ThreadBudget {
    /// A count above the available parallelism runs on that many, which bounds the cache.
    pub(crate) fn new(threads: Option<isize>) -> PyResult<Self> {
        let Some(threads) = threads else {
            return Ok(Self(None));
        };
        let threads = usize::try_from(threads)
            .ok()
            .filter(|&t| t >= 1)
            .ok_or_else(|| {
                PyValueError::new_err(format!("threads must be at least 1; got {threads}"))
            })?;
        let max = std::thread::available_parallelism().map_or(1, usize::from);
        let threads = threads.min(max);
        let mut pools = POOLS.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(&pool) = pools.get(&threads) {
            return Ok(Self(Some(pool)));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let pool: &'static ThreadPool = Box::leak(Box::new(pool));
        pools.insert(threads, pool);
        Ok(Self(Some(pool)))
    }

    pub(crate) fn run<R: Send>(self, f: impl FnOnce() -> R + Send) -> R {
        match self.0 {
            None => f(),
            Some(pool) => pool.install(f),
        }
    }
}
