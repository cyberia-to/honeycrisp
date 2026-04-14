use crate::tape::Tape;
use crate::MemError;

/// Memory layout for inference: weights, scratch, history.
///
/// Three tapes, three lifecycles:
/// - weights: load once, never clear
/// - scratch: clear every token (forward pass temporaries)
/// - history: clear every conversation (accumulated context)
pub struct Layout {
    weights: Tape,
    scratch: Tape,
    history: Tape,
}

impl Layout {
    /// Create inference memory layout.
    ///
    /// - `weights_size`: model weights (e.g. 4GB for 7B q4)
    /// - `scratch_size`: per-token temporaries (e.g. 64MB)
    /// - `history_size`: conversation context (e.g. 512MB for 8K tokens)
    pub fn new(
        weights_size: usize,
        scratch_size: usize,
        history_size: usize,
    ) -> Result<Self, MemError> {
        Ok(Layout {
            weights: Tape::start_warm(weights_size)?,
            scratch: Tape::start(scratch_size)?,
            history: Tape::start(history_size)?,
        })
    }

    /// Weights tape — load model here. Never clear during inference.
    #[inline]
    pub fn weights(&self) -> &Tape {
        &self.weights
    }

    /// Scratch tape — per-token temporaries. Clear after each token.
    #[inline]
    pub fn scratch(&self) -> &Tape {
        &self.scratch
    }

    /// History tape — conversation context. Clear on new conversation.
    #[inline]
    pub fn history(&self) -> &Tape {
        &self.history
    }

    /// Clear scratch only. Call after each token.
    #[inline]
    pub fn clear_pass(&self) {
        self.scratch.clear();
    }

    /// Clear history and scratch. Call on new conversation.
    #[inline]
    pub fn clear_talk(&self) {
        self.history.clear();
        self.scratch.clear();
    }

    /// Total bytes across all tapes.
    pub fn total(&self) -> usize {
        self.weights.total() + self.scratch.total() + self.history.total()
    }

    /// Statistics.
    pub fn stat(&self) -> Stat {
        Stat {
            weights_used: self.weights.used(),
            weights_total: self.weights.total(),
            scratch_used: self.scratch.used(),
            scratch_total: self.scratch.total(),
            history_used: self.history.used(),
            history_total: self.history.total(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Stat {
    pub weights_used: usize,
    pub weights_total: usize,
    pub scratch_used: usize,
    pub scratch_total: usize,
    pub history_used: usize,
    pub history_total: usize,
}

impl std::fmt::Display for Stat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "weights: {}/{} MB | scratch: {}/{} MB | history: {}/{} MB",
            self.weights_used >> 20,
            self.weights_total >> 20,
            self.scratch_used >> 20,
            self.scratch_total >> 20,
            self.history_used >> 20,
            self.history_total >> 20,
        )
    }
}
