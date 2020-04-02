use super::puct::{PUCTPolicy, PUCTSettings, PUCT};
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
use ndarray::Ix3;
use std::fmt;
use std::sync::Arc;
use std::sync::{atomic::AtomicBool, RwLock};
use tokio::sync::mpsc;

/// MuZero policy
pub struct MuzPolicy<G>
where
    G: game::Feature + 'static,
{
    player: G::Player,
    /// PUCT policy instance. Can be taken to gather statistics.
    pub mcts: Option<PUCTPolicy<Simulated<G>>>,
    config: Muz,
}

#[async_trait]
impl<G> MultiplayerPolicy<G> for MuzPolicy<G>
where
    G: game::Feature + 'static,
{
    async fn play(&mut self, board: &G) -> G::Move {
        let net_output = representation(
            self.config.channels.representation.clone(),
            self.config.repr_dimension,
            &board.state_to_feature(self.player),
        )
        .await;

        let simulator = Simulated::new(
            board.turn(),
            net_output,
            board.possible_moves(),
            self.config.channels.dynamics.clone(),
        );

        let mcts_policy_builder = PUCT {
            prediction_channel: self.config.channels.prediction.clone(),
            config: self.config.puct,
            N_PLAYOUTS: self.config.N_PLAYOUTS,
        };

        let mut mcts_policy = mcts_policy_builder.create(self.player);

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
    /// PUCT settings.
    pub puct: PUCTSettings,
    /// Number of PUCT playouts per move.
    pub N_PLAYOUTS: usize,
    /// Evaluation channels
    pub channels: MuzEvaluatorChannels,
    /// Representation board dimension
    pub repr_dimension: Ix3,
}

impl fmt::Display for Muz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MUZ")?;
        writeln!(f, "||{:?}", self.puct)
    }
}

impl<G> MultiplayerPolicyBuilder<G> for Muz
where
    G: game::Feature + 'static,
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
pub struct MuZeroConfig<R, B, A> {
    /// Settings for PUCT search.
    pub puct: PUCTSettings,
    /// Models base directory location.
    pub networks_path: String,
    /// Virtual board shape dimensions.
    pub repr_board_shape: R,
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
pub struct MuzEvaluators<R, B, A> {
    config: MuZeroConfig<R, B, A>,
    prediction_tensorflow: tf::ThreadSafeModel,
    dynamics_tensorflow: tf::ThreadSafeModel,
    representation_tensorflow: tf::ThreadSafeModel,
    channels: MuzEvaluatorChannels,
}

impl<R, B, A> Clone for MuzEvaluators<R, B, A>
where
    R: Dimension,
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

impl<R, B, A> MuzEvaluators<R, B, A>
where
    R: Dimension,
    B: Dimension,
    A: Dimension,
{
    /// Create a new evaluator manager, loading the models and watching
    /// the files if necessary.
    /// If `spawn_tensorflow` is set, also spawn evaluators for the current
    /// channels.
    pub fn new(config: MuZeroConfig<R, B, A>, spawn_tensorflow: bool) -> Self {
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
        let repr_size = self.config.repr_board_shape.size();

        tokio::spawn(prediction_task(
            self.config.batch_size,
            repr_size,
            action_size,
            if self.config.puct.DECODE_VALUE {
                settings::SUPPORT_SHAPE as usize
            } else {
                1
            },
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
            if self.config.puct.DECODE_VALUE {
                settings::SUPPORT_SHAPE as usize
            } else {
                1
            },
            self.dynamics_tensorflow.clone(),
            muz_dyn_rx,
        ));
    }
}
