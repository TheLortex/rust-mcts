// Policies
/// Default number of playouts in a playout-based policy.
pub const DEFAULT_N_PLAYOUTS: usize = 200;
/// Size of the history for PUCT.
pub const DEFAULT_N_HISTORY_PUCT: usize = 2;

// Breakthrough
/// Size of the breakthrough board.
pub const BREAKTHROUGH_K: usize = 5;
/// MuZero hidden state shape for 5x5 breakthrough.
pub const MUZ_BT_SHAPE: (usize, usize, usize) = (5, 5, 16);


/// Range for the categorical value representation.
pub const SUPPORT_SIZE: isize = 1;
/// Resulting shape for the categorical value representation.
pub const SUPPORT_SHAPE: isize = 2 * SUPPORT_SIZE + 1;

// Batched GPU evaluator.

/// Batch size to send to the GPU (PUCT)
pub const GPU_BATCH_SIZE: usize = 4;
/// Number of threads sending batches to the GPU
pub const GPU_N_EVALUATORS: usize = 2;
/// Number of threads generating games, per evaluator
pub const GPU_N_GENERATORS: usize = 4;

/*
pub const GPU_BATCH_SIZE: usize   = 1;
pub const GPU_N_EVALUATORS: usize = 1;
pub const GPU_N_GENERATORS: usize = 1;
*/
