use once_cell::sync::OnceCell;

use crate::BuildError;

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

pub(crate) type BuildResult<T, E = BuildError> = Result<T, BuildFailure<E>>;

#[derive(Clone, Copy, Default)]
pub(crate) struct BuildContext<'a> {
    pub(crate) decision: Option<&'a OnceCell<bool>>,
}

impl BuildContext<'_> {
    pub(crate) fn before_factorization<E>(self) -> BuildResult<(), E> {
        match self.decision {
            Some(decision) if !*decision.wait() => Err(BuildFailure::Cancelled),
            _ => Ok(()),
        }
    }
}

pub(crate) struct CancelOnDrop<'a> {
    pub(crate) decision: &'a OnceCell<bool>,
}

impl Drop for CancelOnDrop<'_> {
    fn drop(&mut self) {
        let _ = self.decision.set(false);
    }
}

/// Erase cancellation only for construction explicitly run without a gate.
pub(crate) fn unrestricted<T, E>(result: BuildResult<T, E>) -> Result<T, E> {
    match result {
        Ok(value) => Ok(value),
        Err(BuildFailure::Failed(error)) => Err(error),
        Err(BuildFailure::Cancelled) => unreachable!("an unrestricted build cannot cancel"),
    }
}
