//! Settings: configuration file definitions and utilities.

use serde_derive::Deserialize;

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "kind")]
/// Possible games and their associated options.
pub enum Game {
    /// Breakthrough
    Breakthrough {
        /// History length.
        history: Option<usize>,
        /// Board size.
        size: usize,
    },
    /// OpenAI Gym
    Gym {
        /// History length.
        history: Option<usize>,
        /// Gym game name.
        name: String,
        /// Gym executor remote address.
        #[serde(default = "default_remote")]
        remote: String,
    },
}

fn default_remote() -> String {
    "localhost:1337".into()
}

impl Game {
    /// Game display name.
    pub fn name(&self) -> String {
        match self {
            Game::Breakthrough { size, .. } => format!("breakthrough-{}", size),
            Game::Gym { name, .. } => format!("gym-{}", name),
        }
    }

    /// Get history for game.
    pub fn history(&self) -> Option<usize> {
        match self {
            Game::Breakthrough { history, .. } => *history,
            Game::Gym { history, .. } => *history,
        }
    }
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// Self-play settings.
pub struct SelfPlay {
    /// GPU batch size.
    pub batch_size: usize,
    /// Number of evaluators: tasks that send batches to GPUs.
    pub evaluators: usize,
    /// Number of generators: tasks that generate games.
    pub generators: usize,
}

const DEFAULT_PLAYOUTS: usize = 200;

/* Standard policies */
#[derive(Deserialize, Copy, Clone, Debug)]
/// MCTS-based policies settings.
pub struct MCTS {
    /// Number of playouts per turn.
    pub playouts: usize,
}

impl Default for MCTS {
    fn default() -> Self {
        Self {
            playouts: DEFAULT_PLAYOUTS,
        }
    }
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// RAVE settings.
pub struct RAVE {
    #[serde(default = "default_uct")]
    /// UCT weight.
    pub uct_weight: f32,
    /// Number of playouts per turn.
    pub playouts: usize,
}

impl Default for RAVE {
    fn default() -> Self {
        Self {
            uct_weight: default_uct(),
            playouts: DEFAULT_PLAYOUTS,
        }
    }
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// UCT settings.
pub struct UCT {
    #[serde(default = "default_uct")]
    /// UCT weight.
    pub uct_weight: f32,
    /// Number of playouts per turn.
    pub playouts: usize,
}

impl Default for UCT {
    fn default() -> Self {
        Self {
            uct_weight: default_uct(),
            playouts: DEFAULT_PLAYOUTS,
        }
    }
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// Flat UCB Monte Carlo settings.
pub struct FlatUCBMonteCarlo {
    /// Number of playouts per turn.
    pub playouts: usize,
    #[serde(default = "default_uct")]
    /// UCB weight
    pub ucb_weight: f32,
}

impl Default for FlatUCBMonteCarlo {
    fn default() -> Self {
        Self {
            ucb_weight: default_uct(),
            playouts: DEFAULT_PLAYOUTS,
        }
    }
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// Flat Monte Carlo settings.
pub struct FlatMonteCarlo {
    /// Number of playouts per turn.
    pub playouts: usize,
}

impl Default for FlatMonteCarlo {
    fn default() -> Self {
        Self {
            playouts: DEFAULT_PLAYOUTS,
        }
    }
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// PPA settings.
pub struct PPA {
    #[serde(default = "default_uct")]
    /// Weight for UCT formula.
    pub uct_weight: f32,
    /// Total number of playouts at each step.
    pub playouts: usize,
    /// α value used in policy gradient.
    pub alpha: f32,
}

impl Default for PPA {
    fn default() -> Self {
        Self {
            uct_weight: default_uct(),
            playouts: DEFAULT_PLAYOUTS,
            alpha: 0.1,
        }
    }
}

fn default_uct() -> f32 {
    0.4
}

#[derive(Deserialize, Copy, Clone, Debug, Default)]
/// Policies settings node.
pub struct Policies {
    #[serde(default)]
    /// RAVE settings
    pub rave: RAVE,
    #[serde(default)]
    /// PPA settings
    pub ppa: PPA,
    #[serde(default)]
    /// Flat Monte Carlo settings
    pub flat: FlatMonteCarlo,
    #[serde(default)]
    /// Flat UCB Monte Carlo settings
    pub flat_ucb: FlatUCBMonteCarlo,
    #[serde(default)]
    /// UCT settings
    pub uct: UCT,
}
/* DL-based policies */
#[derive(Deserialize, Copy, Clone, Debug)]
/// PUCT settings.
pub struct PUCT {
    /// Reward discount value.
    pub discount: f32,
    /// PUCT formula base. (see Deepmind's paper)
    pub c_base: f32,
    /// PUCT formula init. (bis)
    pub c_init: f32,
    /// Root exploration alpha.
    pub root_dirichlet_alpha: f32,
    /// Root exploration fraction.
    pub root_exploration_fraction: f32,
    /// Value support encoding.
    pub value_support: Option<usize>,
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// AlphaZero settings.
pub struct AlphaZero {
    /// Underlying PUCT policy.
    pub puct: PUCT,
}

#[derive(Deserialize, Copy, Clone, Debug)]
/// MuZero settings.
pub struct MuZero {
    /// Underlying PUCT policy.
    pub puct: PUCT,
    /// Reward support encoding.
    pub reward_support: Option<usize>,
    /// Representation board shape.
    pub repr_shape: ndarray::Ix3,
    /// Number of unroll steps when training.
    pub unroll_steps: usize,
    /// Temporal-difference steps when training.
    pub td_steps: usize,
}

/// Global configuration.
#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    /// Game settings.
    pub game: Game,
    /// Self-play settings.
    pub self_play: SelfPlay,
    /// MCTS settings.
    pub mcts: MCTS,
    /// AlphaZero settings.
    pub alpha: Option<AlphaZero>,
    /// MuZero settings.
    pub mu: Option<MuZero>,
    #[serde(default)]
    /// Policies settings.
    pub policies: Policies,
}

use crate::policies::mcts::{muz::MuZeroConfig, puct::AlphaZeroConfig};
impl Config {
    /// Build an AlphaZeroConfig from the global configuration if possible.
    pub fn get_alphazero<A, B>(
        &self,
        action_shape: A,
        board_shape: B,
    ) -> Option<AlphaZeroConfig<A, B>> {
        if let Some(alpha_config) = self.alpha {
            let model_path = format!("data/alpha-{}/model/", self.game.name());

            let alpha_config = AlphaZeroConfig {
                action_shape,
                board_shape,
                puct: alpha_config.puct,
                network_path: model_path,
                watch_models: true,
                batch_size: self.self_play.batch_size,
                n_playouts: self.mcts.playouts,
            };
            Some(alpha_config)
        } else {
            None
        }
    }
    /// Build a MuZeroConfig from the global configuration if possible.
    pub fn get_muzero<A, B>(&self, action_shape: A, board_shape: B) -> Option<MuZeroConfig<B, A>> {
        if let Some(mu_config) = self.mu {
            let models_path = format!("data/mu-{}/models/", self.game.name());

            let mu_config = MuZeroConfig {
                action_shape,
                board_shape,
                muz: mu_config,
                networks_path: models_path,
                watch_models: true,
                batch_size: self.self_play.batch_size,
                n_playouts: self.mcts.playouts,
            };
            Some(mu_config)
        } else {
            None
        }
    }
}

/// Training methods.
pub enum Method {
    /// MuZero
    MuZero,
    /// AlphaZero
    AlphaZero,
}

impl Method {
    /// Display name for method.
    pub fn name(&self) -> &str {
        match self {
            Method::MuZero => "mu",
            Method::AlphaZero => "alpha",
        }
    }
}

#[derive(Debug, Clone)]
/// Simple error wrapper.
pub struct StrError(pub String);

use std::{error, fmt};

impl fmt::Display for StrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl error::Error for StrError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
