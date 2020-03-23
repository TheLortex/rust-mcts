// Policies
pub const DEFAULT_N_PLAYOUTS: usize = 100;
pub const DEFAULT_N_HISTORY_PUCT: usize = 2;

// Breakthrough
pub const BREAKTHROUGH_K: usize = 5;

// Batched GPU evaluator.
pub const GPU_BATCH_SIZE: usize   = 64;
pub const GPU_N_EVALUATORS: usize = 2;
pub const GPU_N_GENERATORS: usize = 8;

/*
pub const GPU_BATCH_SIZE: usize   = 1;
pub const GPU_N_EVALUATORS: usize = 1;
pub const GPU_N_GENERATORS: usize = 1;
*/