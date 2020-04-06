use super::puct::{PUCTPolicy, PUCT};
use crate::deep::evaluator::{dynamics_task, prediction_task, representation_task};
use crate::deep::evaluator::{
    representation, DynamicsEvaluatorChannel, PredictionEvaluatorChannel,
    RepresentationEvaluatorChannel,
};
use crate::deep::file_manager;
use crate::deep::tf;
use crate::game;
use crate::game::meta::simulated::Simulated;
use crate::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};
use crate::settings;

use async_trait::async_trait;
use ndarray::Dimension;
use std::fmt;
use std::sync::Arc;
use std::sync::{atomic::AtomicBool, RwLock};
use tokio::sync::mpsc;

/// MuZero policy
pub struct MuzPolicy<G>
where
    G: game::Features + 'static,
{
    player: G::Player,
    /// PUCT policy instance. Can be taken to gather statistics.
    pub mcts: Option<PUCTPolicy<Simulated<G>>>,
    config: Muz,
}

#[async_trait]
impl<G> MultiplayerPolicy<G> for MuzPolicy<G>
where
    G: game::Features + 'static,
{
    async fn play(&mut self, board: &G) -> G::Move {
        let net_output = representation(
            self.config.channels.representation.clone(),
            self.config.muz.repr_shape,
            &board.state_to_feature(self.player),
        )
        .await;

        let simulator = Simulated::new(
            board.turn(),
            net_output,
            board.get_features(),
            board.possible_moves(),
            self.config.channels.dynamics.clone(),
            self.config.muz.reward_support.unwrap_or(0),
        );

        let mcts_policy_builder = PUCT {
            prediction_channel: self.config.channels.prediction.clone(),
            config: self.config.muz.puct,
            n_playouts: self.config.n_playouts,
        };

        let mut mcts_policy: PUCTPolicy<Simulated<G>> = mcts_policy_builder.create(self.player);

        let action = mcts_policy.play(&simulator).await;
        self.mcts = Some(mcts_policy);
        action
    }
}

/// Channels that can be used to request inferences from tensorflow.
#[derive(Clone)]
pub struct MuzEvaluatorChannels {
    /// Evaluator for the prediction network.
    pub prediction: mpsc::Sender<PredictionEvaluatorChannel>,
    /// Evaluator for the representation network.
    pub representation: mpsc::Sender<RepresentationEvaluatorChannel>,
    /// Evaluator for the dynamics network.
    pub dynamics: mpsc::Sender<DynamicsEvaluatorChannel>,
}

/// MuZero policy builder.
#[derive(Clone)]
pub struct Muz {
    /// Number of PUCT playouts per move.
    pub n_playouts: usize,
    /// Muz settings.
    pub muz: settings::MuZero,
    /// Evaluation channels
    pub channels: MuzEvaluatorChannels,
}

impl fmt::Display for Muz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MUZ")?;
        writeln!(f, "||N_playouts: {:?}", self.n_playouts)?;
        writeln!(f, "|| {:?}", self.muz)
    }
}

impl<G> MultiplayerPolicyBuilder<G> for Muz
where
    G: game::Features + 'static,
{
    type P = MuzPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        MuzPolicy {
            player: color,
            config: self.clone(),
            mcts: None,
        }
    }
}

/// Global configuration for MuZero setup.
#[derive(Clone)]
pub struct MuZeroConfig<B, A> {
    /// Number of playouts for search.
    pub n_playouts: usize,
    /// Settings for PUCT search.
    pub muz: settings::MuZero,
    /// Models base directory location.
    pub networks_path: String,
    /// Board space dimensions.
    pub board_shape: B,
    /// Action shape dimensions.
    pub action_shape: A,
    /// Watch model for update.
    pub watch_models: bool,
    /// GPU batch size.
    pub batch_size: usize,
}

/// Structure that manages the tensorflow models and
/// the batched evaluator tasks.
pub struct MuzEvaluators<B, A> {
    config: MuZeroConfig<B, A>,
    prediction_tensorflow: tf::ThreadSafeModel,
    dynamics_tensorflow: tf::ThreadSafeModel,
    representation_tensorflow: tf::ThreadSafeModel,
    channels: MuzEvaluatorChannels,
}

impl<B, A> Clone for MuzEvaluators<B, A>
where
    B: Dimension,
    A: Dimension,
{
    fn clone(&self) -> Self {
        let (muz_pred_tx, muz_pred_rx) =
            mpsc::channel::<PredictionEvaluatorChannel>(self.config.batch_size);
        let (muz_repr_tx, muz_repr_rx) =
            mpsc::channel::<RepresentationEvaluatorChannel>(self.config.batch_size);
        let (muz_dyn_tx, muz_dyn_rx) =
            mpsc::channel::<DynamicsEvaluatorChannel>(self.config.batch_size);

        let mut ret = Self {
            config: self.config.clone(),
            prediction_tensorflow: self.prediction_tensorflow.clone(),
            dynamics_tensorflow: self.dynamics_tensorflow.clone(),
            representation_tensorflow: self.representation_tensorflow.clone(),
            channels: MuzEvaluatorChannels {
                prediction: muz_pred_tx,
                representation: muz_repr_tx,
                dynamics: muz_dyn_tx,
            },
        };
        ret.spawn_tensorflow_tasks(muz_repr_rx, muz_pred_rx, muz_dyn_rx);
        ret
    }
}

impl<B, A> MuzEvaluators<B, A>
where
    B: Dimension,
    A: Dimension,
{
    /// Create a new evaluator manager, loading the models and watching
    /// the files if necessary.
    /// If `spawn_tensorflow` is set, also spawn evaluators for the current
    /// channels.
    pub fn new(config: MuZeroConfig<B, A>, spawn_tensorflow: bool) -> MuzEvaluators<B, A> {
        let (muz_pred_tx, muz_pred_rx) =
            mpsc::channel::<PredictionEvaluatorChannel>(config.batch_size);
        let (muz_repr_tx, muz_repr_rx) =
            mpsc::channel::<RepresentationEvaluatorChannel>(config.batch_size);
        let (muz_dyn_tx, muz_dyn_rx) = mpsc::channel::<DynamicsEvaluatorChannel>(config.batch_size);

        let prediction_path = format!("{}{}", config.networks_path, "pv");
        let dynamics_path = format!("{}{}", config.networks_path, "dyn");
        let representation_path = format!("{}{}", config.networks_path, "state");

        let prediction_tensorflow = Arc::new((
            AtomicBool::new(false),
            RwLock::new(tf::load_model(&prediction_path)),
        ));
        let dynamics_tensorflow = Arc::new((
            AtomicBool::new(false),
            RwLock::new(tf::load_model(&dynamics_path)),
        ));
        let representation_tensorflow = Arc::new((
            AtomicBool::new(false),
            RwLock::new(tf::load_model(&representation_path)),
        ));

        let watch_models = config.watch_models;

        let mut ret = Self {
            config,
            prediction_tensorflow,
            dynamics_tensorflow,
            representation_tensorflow,
            channels: MuzEvaluatorChannels {
                prediction: muz_pred_tx,
                representation: muz_repr_tx,
                dynamics: muz_dyn_tx,
            },
        };

        if spawn_tensorflow {
            ret.spawn_tensorflow_tasks(muz_repr_rx, muz_pred_rx, muz_dyn_rx);
        }
        if watch_models {
            ret.spawn_file_watchers();
        }
        ret
    }

    /// Get evaluation requests sender channels to give to Muz.
    /// Useless if tensorflow processes hasn't been started.
    pub fn get_channels(&self) -> MuzEvaluatorChannels {
        self.channels.clone()
    }

    fn spawn_file_watchers(&self) {
        let prediction_path = format!("{}{}", self.config.networks_path, "pv");
        let dynamics_path = format!("{}{}", self.config.networks_path, "dyn");
        let representation_path = format!("{}{}", self.config.networks_path, "state");

        file_manager::watch_model(self.prediction_tensorflow.clone(), &prediction_path);
        file_manager::watch_model(self.dynamics_tensorflow.clone(), &dynamics_path);
        file_manager::watch_model(self.representation_tensorflow.clone(), &representation_path);
    }

    fn spawn_tensorflow_tasks(
        &mut self,
        muz_repr_rx: mpsc::Receiver<RepresentationEvaluatorChannel>,
        muz_pred_rx: mpsc::Receiver<PredictionEvaluatorChannel>,
        muz_dyn_rx: mpsc::Receiver<DynamicsEvaluatorChannel>,
    ) {
        let board_size = self.config.board_shape.size();
        let action_size = self.config.action_shape.size();
        let repr_size = self.config.muz.repr_shape.size();

        tokio::spawn(prediction_task(
            self.config.batch_size,
            repr_size,
            action_size,
            2 * self.config.muz.puct.value_support.unwrap_or(0) + 1,
            self.prediction_tensorflow.clone(),
            muz_pred_rx,
            None,
        ));

        tokio::spawn(representation_task(
            self.config.batch_size,
            board_size,
            repr_size,
            self.representation_tensorflow.clone(),
            muz_repr_rx,
        ));

        tokio::spawn(dynamics_task(
            self.config.batch_size,
            repr_size,
            action_size,
            2 * self.config.muz.reward_support.unwrap_or(0) + 1,
            self.dynamics_tensorflow.clone(),
            muz_dyn_rx,
        ));
    }
}
