//! Pilot decision and blocking gate before speculative Schwarz factorization.

use once_cell::sync::OnceCell;

use crate::BuildError;

/// A cancellation is neither a failed design nor a successful build without a target.
#[derive(Debug)]
pub(crate) enum BuildFailure<E = BuildError> {
    Cancelled,
    Failed(E),
}

impl<E> From<E> for BuildFailure<E> {
    fn from(error: E) -> Self {
        Self::Failed(error)
    }
}

impl<E> BuildFailure<E> {
    pub(crate) fn map<F>(self, f: impl FnOnce(E) -> F) -> BuildFailure<F> {
        match self {
            Self::Cancelled => BuildFailure::Cancelled,
            Self::Failed(error) => BuildFailure::Failed(f(error)),
        }
    }
}

pub(crate) type BuildResult<T, E = BuildError> = Result<T, BuildFailure<E>>;

/// Stages are also the deterministic test seams for cancellation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BuildStage {
    Factor,
    #[cfg(test)]
    Factorization,
}

/// One pilot owns the decision; all of its build workers observe the same terminal state.
#[derive(Default)]
pub(crate) struct BuildControl {
    // Empty permits preparation; true releases factorization; false cancels the build.
    decision: OnceCell<bool>,
    #[cfg(test)]
    hook: Option<Box<dyn Fn(BuildStage) + Send + Sync>>,
}

impl BuildControl {
    pub(crate) fn finish(&self, required: bool) {
        // The first decision is terminal, including when the cancellation guard drops.
        let _ = self.decision.set(required);
    }

    pub(crate) fn guard(&self) -> CancelOnDrop<'_> {
        CancelOnDrop(self)
    }

    #[cfg(test)]
    pub(crate) fn with_hook(hook: impl Fn(BuildStage) + Send + Sync + 'static) -> Self {
        Self {
            hook: Some(Box::new(hook)),
            ..Self::default()
        }
    }
}

/// Drop this inside the scoped thread closure, before the scope joins its children.
pub(crate) struct CancelOnDrop<'a>(&'a BuildControl);

impl Drop for CancelOnDrop<'_> {
    fn drop(&mut self) {
        self.0.finish(false);
    }
}

/// Unrestricted callers keep their existing interfaces and cannot be cancelled.
#[derive(Clone, Copy, Default)]
pub(crate) struct BuildContext<'a>(pub(crate) Option<&'a BuildControl>);

impl BuildContext<'_> {
    #[inline]
    pub(crate) fn checkpoint<E>(self, stage: BuildStage) -> BuildResult<(), E> {
        let Some(control) = self.0 else {
            return Ok(());
        };
        #[cfg(test)]
        if let Some(hook) = &control.hook {
            hook(stage);
        }
        #[cfg(not(test))]
        let _ = stage;
        if control.decision.get() == Some(&false) {
            Err(BuildFailure::Cancelled)
        } else {
            Ok(())
        }
    }

    /// The backend has no cancellation API: do not enter it until Schwarz is needed.
    pub(crate) fn before_factorization<E>(self) -> BuildResult<(), E> {
        self.checkpoint(BuildStage::Factor)?;
        if let Some(control) = self.0 {
            if !*control.decision.wait() {
                return Err(BuildFailure::Cancelled);
            }
        }
        Ok(())
    }
}

/// Only unrestricted wrappers may erase the internal cancellation outcome.
pub(crate) fn unrestricted<T, E>(result: BuildResult<T, E>) -> Result<T, E> {
    match result {
        Ok(value) => Ok(value),
        Err(BuildFailure::Failed(error)) => Err(error),
        Err(BuildFailure::Cancelled) => unreachable!("an unrestricted build cannot cancel"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn every_waiter_is_released_by_either_decision() {
        for required in [false, true] {
            let control = Arc::new(BuildControl::default());
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        let control = Arc::clone(&control);
                        let started_tx = started_tx.clone();
                        scope.spawn(move || {
                            started_tx.send(()).unwrap();
                            BuildContext(Some(&control)).before_factorization::<BuildError>()
                        })
                    })
                    .collect();
                for _ in 0..4 {
                    started_rx.recv().unwrap();
                }
                control.finish(required);
                control.finish(!required); // The first decision is terminal.
                for handle in handles {
                    let result = handle.join().unwrap();
                    assert_eq!(result.is_ok(), required);
                }
            });
        }
    }

    #[test]
    fn unwinding_cancels_before_the_scope_joins() {
        let control = BuildControl::default();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            std::thread::scope(|scope| {
                let _guard = control.guard();
                scope.spawn(|| {
                    assert!(matches!(
                        BuildContext(Some(&control)).before_factorization::<BuildError>(),
                        Err(BuildFailure::Cancelled)
                    ));
                });
                panic!("pilot panic");
            });
        }));
        assert!(panic.is_err());
    }
}
