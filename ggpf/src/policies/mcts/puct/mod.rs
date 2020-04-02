use crate::deep::evaluator::{prediction, prediction_task, PredictionEvaluatorChannel};
use crate::deep::file_manager;
use crate::deep::tf;
use crate::game;
use crate::policies::mcts::{BaseMCTSPolicy, MCTSTreeNode, WithMCTSPolicy};
use crate::policies::MultiplayerPolicyBuilder;
use crate::settings;

use async_trait::async_trait;
use ndarray::Array;
use ndarray::Dimension;
use rand_distr::{Distribution, Gamma};
use std::collections::HashMap;
use std::f32;
use std::fmt;
use std::iter::*;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

/// PUCT move statistics.
#[derive(Debug, Clone, Copy)]
pub struct PUCTMoveInfo {
    /// Value (expected discounted reward for move, relative to current player)
    pub Q: f32,
    /// Number of times the move has been explored.
    pub N_a: f32,
    /// Move policy as predicted by the network.
    pub pi: f32,
    /// Immediate reward yielded by move.
    pub reward: f32,
}

/// PUCT node statistics.
#[derive(Debug, Clone, Copy)]
pub struct PUCTNodeInfo {
    /// Node visit count.
    pub count: f32,
}
/**
 * The game state evaluator
 */
pub trait Evaluator<G: game::Features> = Fn(<G as game::Game>::Player, &G) -> (Array<f32, <G as game::Features>::ActionDim>, f32)
    + Send
    + Sync;

/**
 * Common PUCT
 */
#[derive(Copy, Clone, fmt::Debug)]
pub struct PUCTSettings {
    /// PUCT formule base constant.
    pub C_BASE: f32,
    /// PUCT formula init constant.
    pub C_INIT: f32,
    /// Dirichlet noise alpha.
    pub ROOT_DIRICHLET_ALPHA: f32,
    /// Dirichlet noise fraction.
    pub ROOT_EXPLORATION_FRACTION: f32,
    /// Value discounting.
    pub DISCOUNT: f32,
    /// Decode categorical encoding for value.
    pub DECODE_VALUE: bool,
}

/**
 * Default inspired from AlphaZero paper.
 */
impl Default for PUCTSettings {
    fn default() -> Self {
        Self {
            C_BASE: 19652.,
            C_INIT: 1.25,
            ROOT_DIRICHLET_ALPHA: 0.3,
            ROOT_EXPLORATION_FRACTION: 0.25,
            DISCOUNT: 0.997,
            DECODE_VALUE: false,
        }
    }
}

/// PUCT policy.
#[derive(Clone)]
pub struct PUCTPolicy_<G>
where
    G: game::Features,
{
    color: G::Player,
    config: PUCTSettings,
    prediction_channel: mpsc::Sender<PredictionEvaluatorChannel>,
    min_tree: f32,
    max_tree: f32,
}

impl<G> PUCTPolicy_<G>
where
    G: game::Features + super::MCTSGame,
{
    fn normalize(&self, x: f32) -> f32 {
        if self.min_tree < self.max_tree {
            (x - self.min_tree) / (self.max_tree - self.min_tree)
        } else {
            x
        }
    }
}

type PUCTPlayoutInfo<G> = (
    Option<HashMap<<G as game::Base>::Move, f32>>,
    f32,
    <G as game::Game>::Player,
);

#[allow(clippy::float_cmp)]
#[async_trait]
impl<G> BaseMCTSPolicy<G> for PUCTPolicy_<G>
where
    G: game::Features + super::MCTSGame,
{
    type NodeInfo = PUCTNodeInfo;
    type MoveInfo = PUCTMoveInfo;
    type PlayoutInfo = PUCTPlayoutInfo<G>;

    fn get_value(
        &self,
        _board: &G,
        _action: &G::Move,
        node_info: &Self::NodeInfo,
        move_info: &Self::MoveInfo,
        exploration: bool,
    ) -> f32 {
        if exploration {
            let N = node_info.count;
            let v = move_info;
            let pb_c =
                ((N + self.config.C_BASE + 1.) / self.config.C_BASE).ln() + self.config.C_INIT;
            let prior = pb_c * v.pi * (N.sqrt() / (v.N_a + 1.));
            let value = self.normalize(move_info.reward + self.config.DISCOUNT * move_info.Q);
            prior + value
        } else {
            move_info.N_a
        }
    }

    fn default_move(&self, _board: &G, _action: &G::Move) -> Self::MoveInfo {
        PUCTMoveInfo {
            Q: 0.,
            N_a: 0.,
            pi: 1.,
            reward: 0.,
        }
    }

    fn default_node(&self, _board: &G) -> Self::NodeInfo {
        PUCTNodeInfo { count: 0. }
    }

    fn backpropagate(
        &mut self,
        leaf: Arc<RwLock<MCTSTreeNode<G, Self>>>,
        _history: &[G::Move],
        (policy, mut value, pov): Self::PlayoutInfo,
    ) {
        // todo: assert leaf.turn == pov
        // assert_eq!(leaf.borrow().info.state.turn(), *pov);

        if let Some(mut policy) = policy {
            // save probabilities of newly created node.
            let mut leaf = leaf.write().unwrap();
            if leaf.parent.is_none() {
                // root node: add dirichlet noise.
                let frac = self.config.ROOT_EXPLORATION_FRACTION;
                let gamma = Gamma::new(self.config.ROOT_DIRICHLET_ALPHA, 1.0).unwrap();
                for (_, val) in policy.iter_mut() {
                    let noise = gamma.sample(&mut rand::thread_rng());
                    *val = frac * (*val) + (1. - frac) * noise;
                }
            }

            let z: f32 = leaf
                .info
                .moves
                .keys()
                .map(|m| policy.get(&m).unwrap())
                .sum();
            let z = if z == 0. { 1. } else { z };
            for (m, info) in leaf.info.moves.iter_mut() {
                info.pi = policy.get(&m).unwrap() / z;
            }
        }

        value = -leaf.read().unwrap().info.reward + self.config.DISCOUNT * value;

        let mut tree_position = leaf;
        while tree_position.read().unwrap().parent.is_some() {
            // extract child
            let (tree_pointer, action) = tree_position
                .read()
                .unwrap()
                .parent
                .as_ref()
                .map(|(t, a)| (t.upgrade().unwrap(), *a))
                .unwrap();

            tree_position = tree_pointer;

            let mut tree_node = tree_position.write().unwrap();

            let relative_value = if tree_node.info.state.turn() == pov {
                value
            } else {
                -value
            };

            tree_node.info.node.count += 1.;

            let node_reward = tree_node
                .moves
                .get_mut(&action)
                .unwrap()
                .read()
                .unwrap()
                .info
                .reward;

            let mut v = tree_node.info.moves.get_mut(&action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (relative_value - (*v).Q) / (*v).N_a;
            (*v).reward = node_reward;

            if (*v).Q < self.min_tree {
                self.min_tree = (*v).Q
            }

            if (*v).Q > self.max_tree {
                self.max_tree = (*v).Q
            }

            value = if tree_node.info.state.turn() == pov {
                tree_node.info.reward
            } else {
                -tree_node.info.reward
            } + self.config.DISCOUNT * value;
        }
    }
    /*

        fn backpropagate(
            &self,
            _index: usize,
            info: &mut MCTSNode<G, Self>,
            action: &G::Move,
            history: &[G::Move],
            (policy, value, pov): &Self::PlayoutInfo,
        ) {
            let value = if info.state.turn() == *pov {
                *value
            } else {
                1. - *value
            };

            info.node.count += 1.;

            let mut v = info.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (value - (*v).Q) / (*v).N_a;
        }
    */
    async fn simulate(&self, board: &G) -> Self::PlayoutInfo {
        if !board.is_finished() {
            // NN predicts a good policy for current player + expectation of winning from this state.
            let (policy, value) = prediction(
                self.prediction_channel.clone(),
                board.turn(),
                board,
                self.config.DECODE_VALUE,
            )
            .await;
            let policy = board.feature_to_moves(&policy);
            (Some(policy), value, board.turn())
        } else {
            (None, 0., board.turn())
        }
    }
}

/**
 *  PUCT policy built from MCTS description.
 */
pub type PUCTPolicy<G> = WithMCTSPolicy<G, PUCTPolicy_<G>>;

/// PUCT policy builder
#[derive(Clone)]
pub struct PUCT {
    /// PUCT configuration.
    pub config: PUCTSettings,
    /// Number of playouts.
    pub N_PLAYOUTS: usize,
    /// State evaluation function.
    pub prediction_channel: mpsc::Sender<PredictionEvaluatorChannel>,
}

impl fmt::Display for PUCT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BATCHED PUCT")?;
        writeln!(f, "||{:?}", self.config)
    }
}

impl<G> MultiplayerPolicyBuilder<G> for PUCT
where
    G: game::Features + super::MCTSGame,
{
    type P = PUCTPolicy<G>;

    fn create(&self, color: G::Player) -> PUCTPolicy<G> {
        WithMCTSPolicy::new(
            PUCTPolicy_::<G> {
                color,
                config: self.config,
                prediction_channel: self.prediction_channel.clone(),
                min_tree: f32::MAX,
                max_tree: -f32::MAX,
            },
            self.N_PLAYOUTS,
        )
    }
}

/// Global configuration for AlphaZero setup.
#[derive(Clone)]
pub struct AlphaZeroConfig<A, B> {
    /// Settings for PUCT search.
    pub puct: PUCTSettings,
    /// Model directory location.
    pub network_path: String,
    /// Board space dimensions.
    pub board_shape: B,
    /// Action shape dimensions.
    pub action_shape: A,
    /// Watch model for update.
    pub watch_models: bool,
    /// GPU batch size.
    pub batch_size: usize,
}

/// Structure that manages the tensorflow model and
/// the batched evaluator task.
pub struct AlphaZeroEvaluators<B, A> {
    config: AlphaZeroConfig<B, A>,
    prediction_tensorflow: tf::ThreadSafeModel,
    channel: mpsc::Sender<PredictionEvaluatorChannel>,
}

impl<B, A> Clone for AlphaZeroEvaluators<B, A>
where
    B: Dimension,
    A: Dimension,
{
    fn clone(&self) -> Self {
        let (alpha_pred_tx, alpha_pred_rx) =
            mpsc::channel::<PredictionEvaluatorChannel>(2 * self.config.batch_size);

        let mut ret = Self {
            config: self.config.clone(),
            prediction_tensorflow: self.prediction_tensorflow.clone(),
            channel: alpha_pred_tx,
        };
        ret.spawn_tensorflow_task(alpha_pred_rx);
        ret
    }
}

impl<B, A> AlphaZeroEvaluators<B, A>
where
    B: Dimension,
    A: Dimension,
{
    /// Create a new evaluator manager, loading the models and watching
    /// the files if necessary.
    /// If `spawn_tensorflow` is set, also spawn evaluators for the current
    /// channels.
    pub fn new(config: AlphaZeroConfig<B, A>, spawn_tensorflow: bool) -> Self {
        let (alpha_pred_tx, alpha_pred_rx) =
            mpsc::channel::<PredictionEvaluatorChannel>(2 * config.batch_size);

        let prediction_tensorflow = Arc::new((
            AtomicBool::new(false),
            RwLock::new(tf::load_model(&config.network_path)),
        ));
        let watch_models = config.watch_models;

        let mut ret = Self {
            config,
            prediction_tensorflow,
            channel: alpha_pred_tx,
        };

        if spawn_tensorflow {
            ret.spawn_tensorflow_task(alpha_pred_rx);
        }
        if watch_models {
            ret.spawn_file_watcher();
        }
        ret
    }

    /// Get evaluation requests sender channel to give to PUCT.
    /// Useless if tensorflow processes hasn't been started.
    pub fn get_channel(&self) -> mpsc::Sender<PredictionEvaluatorChannel> {
        self.channel.clone()
    }

    fn spawn_file_watcher(&self) {
        file_manager::watch_model(
            self.prediction_tensorflow.clone(),
            &self.config.network_path,
        );
    }

    fn spawn_tensorflow_task(&mut self, alpha_pred_rx: mpsc::Receiver<PredictionEvaluatorChannel>) {
        let board_size = self.config.board_shape.size();
        let action_size = self.config.action_shape.size();

        tokio::spawn(prediction_task(
            self.config.batch_size,
            board_size,
            action_size,
            if self.config.puct.DECODE_VALUE {
                settings::SUPPORT_SHAPE as usize
            } else {
                1
            },
            self.prediction_tensorflow.clone(),
            alpha_pred_rx,
            None,
        ));
    }
}
