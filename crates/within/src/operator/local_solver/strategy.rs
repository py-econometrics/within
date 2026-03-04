/// Determines how the local Gramian solve is performed for a subdomain.
#[derive(Debug, Clone)]
pub enum LocalSolveStrategy {
    /// The local Gramian is naturally a graph Laplacian (no augmentation needed).
    Laplacian,

    /// The local Gramian is SDDM but not Laplacian, so Gremban augmentation added
    /// an extra node.
    Sddm,

    /// Factor-pair domain where the bipartite Gramian maps to a Laplacian via
    /// sign-flipping the second block. No augmentation was needed.
    LaplacianGramian { first_block_size: usize },

    /// Factor-pair domain where the Gramian needed Gremban augmentation.
    GramianAugmented { first_block_size: usize },
}

impl LocalSolveStrategy {
    /// Map a bipartite hint and augmentation flag to the appropriate local
    /// solve strategy.
    pub fn from_flags(first_block_size: Option<usize>, was_augmented: bool) -> Self {
        match first_block_size {
            Some(fbs) => {
                if was_augmented {
                    Self::GramianAugmented {
                        first_block_size: fbs,
                    }
                } else {
                    Self::LaplacianGramian {
                        first_block_size: fbs,
                    }
                }
            }
            None => {
                if was_augmented {
                    Self::Sddm
                } else {
                    Self::Laplacian
                }
            }
        }
    }
}
