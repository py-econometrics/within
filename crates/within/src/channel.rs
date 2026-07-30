//! Coefficient-column addressing, shared by the design, the error payloads and
//! the public solve result.

/// One coefficient column of a term; `column` indexes the per-level block, intercept first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Channel {
    /// Index into the design's term list.
    pub term: usize,
    /// Column within the term's per-level block.
    pub column: usize,
}

/// A cross-factor channel pair: one Gramian cross-block per pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelPair {
    /// The channel indexing the cross-block's rows.
    pub rows: Channel,
    /// The channel indexing the cross-block's columns.
    pub cols: Channel,
}

impl std::fmt::Display for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "term {} column {}", self.term, self.column)
    }
}

impl std::fmt::Display for ChannelPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} and {}", self.rows, self.cols)
    }
}
