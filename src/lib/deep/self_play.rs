use crate::game;

use crate::game::meta::simulated::Simulated;
use crate::game::GameBuilder;
use crate::policies::mcts::puct::PUCT;
use crate::policies::{
    mcts::muz::Muz, mcts::puct::PUCTSettings, MultiplayerPolicy, MultiplayerPolicyBuilder,
};
use crate::settings;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Axis, Dimension, Ix1};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use tensorflow::{Graph, Session};

/**
 *  Asynchronous evaluation functions.
 *
 *  For each kind of evaluation there is an evaluator that can be
 *  provided to PUCT/Muz and an associated evaluator task that sends
 *  the request to the GPU while batching them.
 */
use super::evaluator::{
    DynamicsEvaluatorChannel, PredictionEvaluatorChannel, RepresentationEvaluatorChannel,
};

/**
 *  Game history data generated from self-play
 */
pub struct GameHistoryEntry<G>
where
    G: game::Feature,
{
    /// List of board states, except for the final state.
    pub state: Array<f32, <G::StateDim as Dimension>::Larger>,
    /// MCTS exploration statistics for the root node of the curent policy.
    pub policy: Array<f32, <G::ActionDim as Dimension>::Larger>,
    /// One-hot encoding of the action taken by the policy.
    pub action: Array<f32, <G::ActionDim as Dimension>::Larger>,
    /// Value estimation of the root node.
    pub value: Array<f32, Ix1>,
    /// Reward obtained after performing the action.
    pub reward: Array<f32, Ix1>,
    /// Whose turn.
    pub turn: Vec<f32>,
}

//  /$$      /$$ /$$   /$$ /$$$$$$$$ /$$$$$$$$ /$$$$$$$   /$$$$$$
// | $$$    /$$$| $$  | $$|_____ $$ | $$_____/| $$__  $$ /$$__  $$
// | $$$$  /$$$$| $$  | $$     /$$/ | $$      | $$  \ $$| $$  \ $$
// | $$ $$/$$ $$| $$  | $$    /$$/  | $$$$$   | $$$$$$$/| $$  | $$
// | $$  $$$| $$| $$  | $$   /$$/   | $$__/   | $$__  $$| $$  | $$
// | $$\  $ | $$| $$  | $$  /$$/    | $$      | $$  \ $$| $$  | $$
// | $$ \/  | $$|  $$$$$$/ /$$$$$$$$| $$$$$$$$| $$  | $$|  $$$$$$/
// |__/     |__/ \______/ |________/|________/|__/  |__/ \______/
//

/*
 *  The game generator continuously generates self-play games using Muz policies.
 */
fn muzero_game_generator_task<G, GB, H>(
    settings: (PUCTSettings, H),
    game_builder: GB,
    prediction_channel: mpsc::SyncSender<PredictionEvaluatorChannel>,
    dynamics_channel: mpsc::SyncSender<DynamicsEvaluatorChannel>,
    representation_channel: mpsc::SyncSender<RepresentationEvaluatorChannel>,
    output_chan: mpsc::SyncSender<GameHistoryEntry<G>>,
    indicator_bar: Arc<Box<ProgressBar>>,
) where
    G: game::Feature + game::SingleWinner + Send + Sync + 'static,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
    GB: GameBuilder<G>,
    H: Dimension + 'static,
{
    let (puct_settings, hidden_shape) = settings;
    let h1 = hidden_shape.clone();

    let muz = Muz::<G, H> {
        _g: PhantomData,
        _h: PhantomData,
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        puct: puct_settings,
        prediction_evaluate: Arc::new(move |pov: G::Player, board: &Simulated<G, H>| {
            super::evaluator::prediction(prediction_channel.clone(), pov, board.clone(), true)
        }),
        representation_evaluate: Arc::new(move |board: Array<f32, G::StateDim>| {
            super::evaluator::representation(representation_channel.clone(), h1.clone(), board)
        }),
        dynamics_evaluate: Arc::new(move |board, action| {
            super::evaluator::dynamics(
                dynamics_channel.clone(),
                hidden_shape.clone(),
                board.clone(),
                action.clone(),
                true,
            )
        }),
    };

    loop {
        let mut p1 = muz.create(G::players()[0]);
        let mut p2 = muz.create(G::players()[1]);

        let mut state: G =
            game_builder.create(*G::players().choose(&mut rand::thread_rng()).unwrap());

        let mut history_state = vec![];
        let mut history_policy = vec![];
        let mut history_value = vec![];
        let mut history_action = vec![];
        let mut history_reward = vec![];
        let mut history_turn = vec![];

        while { !state.is_finished() } {
            let policy = if state.turn() == G::players()[0] {
                &mut p1
            } else {
                &mut p2
            };
            let action = policy.play(&state);

            /* Save search statistics */
            let mcts = policy.mcts.take().unwrap();
            let game_node = mcts.root.as_ref().unwrap();
            let visit_count = game_node.borrow().info.node.count;

            let monte_carlo_distribution: HashMap<G::Move, f32> = HashMap::from_iter(
                game_node
                    .borrow()
                    .info
                    .moves
                    .iter()
                    .map(|(k, v)| (*k, v.N_a / visit_count)),
            );

            let root_value: f32 = game_node
                .borrow()
                .info
                .moves
                .iter()
                .map(|(_, v)| v.reward + (puct_settings.DISCOUNT * v.Q * v.N_a / visit_count))
                .sum();

            history_turn.push(state.turn().into() as f32);
            history_state.push(state.state_to_feature(state.turn()).insert_axis(Axis(0)));
            history_policy
                .push(G::moves_to_feature(&monte_carlo_distribution).insert_axis(Axis(0)));
            history_value.push(Array::from_elem(ndarray::Ix1(1), root_value));
            history_action.push(G::move_to_feature(action).insert_axis(Axis(0)));

            let reward = state.play(&action);
            history_reward.push(Array::from_elem(ndarray::Ix1(1), reward));
        }

        let history_state_view: Vec<_> = history_state.iter().map(|x| x.view()).collect();
        let history_policy_view: Vec<_> = history_policy.iter().map(|x| x.view()).collect();
        let history_action_view: Vec<_> = history_action.iter().map(|x| x.view()).collect();
        let history_value_view: Vec<_> = history_value.iter().map(|x| x.view()).collect();
        let history_reward_view: Vec<_> = history_reward.iter().map(|x| x.view()).collect();

        output_chan
            .send(GameHistoryEntry {
                state: ndarray::stack(Axis(0), &history_state_view).unwrap(),
                policy: ndarray::stack(Axis(0), &history_policy_view).unwrap(),
                action: ndarray::stack(Axis(0), &history_action_view).unwrap(),
                value: ndarray::stack(Axis(0), &history_value_view).unwrap(),
                reward: ndarray::stack(Axis(0), &history_reward_view).unwrap(),
                turn: history_turn,
            })
            .ok()
            .unwrap();

        indicator_bar.inc(1 as u64);
    }
}

use std::sync::Arc;
use std::sync::RwLock;
use std::thread;

/**
 *  MuZero self-play games generator
 *
 *  Spawn several tasks (number according to settings) that performs self-play games
 *  using the MuZero policy, sending them in the `output_chan` channel.
 *
 *  # Params
 *
 *  - `puct_settings`: configuration for PUCT policy and virtual state dimension.
 *  - `game_builder`: game builder.
 *  - `prediction_tensorflow`: interface for the prediction network.
 *  - `dynamics_tensorflow`: interface for the dynamics network.
 *  - `representation_tensorflow`: interface for the representation network.
 *  - `output_chan`: communication channel to emit the generated games.
 *
 *  # Panics
 *
 *  This function will panic if the evaluator shapes doesn't fit,
 *  or if the CUDA executor goes out of memory.
 */
pub fn muzero_game_generator<G, GB, H>(
    settings: (PUCTSettings, H),
    game_builder: GB,
    prediction_tensorflow: Arc<(AtomicBool, RwLock<(Graph, Session)>)>,
    dynamics_tensorflow: Arc<(AtomicBool, RwLock<(Graph, Session)>)>,
    representation_tensorflow: Arc<(AtomicBool, RwLock<(Graph, Session)>)>,
    output_chan: mpsc::SyncSender<GameHistoryEntry<G>>,
) where
    G: game::Feature + game::SingleWinner + Send + Sync + Clone + Hash + Eq + 'static,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
    GB: GameBuilder<G> + Copy + Sync + Send + 'static,
    H: Dimension + 'static,
{
    let repr_dims = settings.1.clone();

    let indicator_bar = ProgressBar::new_spinner();
    indicator_bar.set_style(
        ProgressStyle::default_spinner()
            .template("[{spinner}] {wide_bar} {pos} games generated ({elapsed_precise})"),
    );
    indicator_bar.enable_steady_tick(200);
    let bar_box = Arc::new(Box::new(indicator_bar));

    let mut join_handles = vec![];
    let mut join_handles_ev = vec![];

    //let gb_box = Arc::new(Box::new(game_builder));

    let state: G = game_builder.create(G::players()[0]);

    let board_size = state.state_dimension().size();
    let action_size = G::action_dimension().size();
    let repr_size = repr_dims.size();

    log::debug!("Representation: {}", repr_size);
    log::debug!("Action: {}", action_size);
    log::debug!("Board: {}", board_size);

    for _ in 0..settings::GPU_N_EVALUATORS {
        let (pred_tx, pred_rx) =
            mpsc::sync_channel::<PredictionEvaluatorChannel>(2 * settings::GPU_BATCH_SIZE);
        let (dyn_tx, dyn_rx) =
            mpsc::sync_channel::<DynamicsEvaluatorChannel>(2 * settings::GPU_BATCH_SIZE);
        let (repr_tx, repr_rx) =
            mpsc::sync_channel::<RepresentationEvaluatorChannel>(2 * settings::GPU_BATCH_SIZE);

        for _ in 0..settings::GPU_N_GENERATORS {
            let pred_tx = pred_tx.clone();
            let dyn_tx = dyn_tx.clone();
            let repr_tx = repr_tx.clone();
            let output_tx = output_chan.clone();
            let czop = bar_box.clone();
            let settings = settings.clone();

            join_handles.push(thread::spawn(move || {
                muzero_game_generator_task(
                    settings,
                    game_builder,
                    pred_tx,
                    dyn_tx,
                    repr_tx,
                    output_tx,
                    czop,
                )
            }));
        }

        let prediction_tensorflow = prediction_tensorflow.clone();
        let representation_tensorflow = representation_tensorflow.clone();
        let dynamics_tensorflow = dynamics_tensorflow.clone();

        join_handles_ev.push(thread::spawn(move || {
            super::evaluator::prediction_task(
                repr_size,
                action_size,
                settings::SUPPORT_SHAPE as usize,
                prediction_tensorflow.as_ref(),
                pred_rx,
            )
        }));

        join_handles_ev.push(thread::spawn(move || {
            super::evaluator::representation_task(
                board_size,
                repr_size,
                representation_tensorflow.as_ref(),
                repr_rx,
            )
        }));

        join_handles_ev.push(thread::spawn(move || {
            super::evaluator::dynamics_task(
                repr_size,
                action_size,
                settings::SUPPORT_SHAPE as usize,
                dynamics_tensorflow.as_ref(),
                dyn_rx,
            )
        }));
    }

    for join_handle in join_handles.drain(..) {
        join_handle.join().unwrap();
    }

    for join_handle in join_handles_ev.drain(..) {
        join_handle.join().unwrap();
    }
}

//   /$$$$$$  /$$       /$$$$$$$  /$$   /$$  /$$$$$$        /$$$$$$$$ /$$$$$$$$ /$$$$$$$   /$$$$$$
//  /$$__  $$| $$      | $$__  $$| $$  | $$ /$$__  $$      |_____ $$ | $$_____/| $$__  $$ /$$__  $$
// | $$  \ $$| $$      | $$  \ $$| $$  | $$| $$  \ $$           /$$/ | $$      | $$  \ $$| $$  \ $$
// | $$$$$$$$| $$      | $$$$$$$/| $$$$$$$$| $$$$$$$$          /$$/  | $$$$$   | $$$$$$$/| $$  | $$
// | $$__  $$| $$      | $$____/ | $$__  $$| $$__  $$         /$$/   | $$__/   | $$__  $$| $$  | $$
// | $$  | $$| $$      | $$      | $$  | $$| $$  | $$        /$$/    | $$      | $$  \ $$| $$  | $$
// | $$  | $$| $$$$$$$$| $$      | $$  | $$| $$  | $$       /$$$$$$$$| $$$$$$$$| $$  | $$|  $$$$$$/
// |__/  |__/|________/|__/      |__/  |__/|__/  |__/      |________/|________/|__/  |__/ \______/
//

/*
 *  The game generator continuously generates self-play games using PUCT policies.
 */
fn alphazero_game_generator_task<G, GB>(
    puct_settings: PUCTSettings,
    game_builder: GB,
    prediction_channel: mpsc::SyncSender<PredictionEvaluatorChannel>,
    output_chan: mpsc::SyncSender<GameHistoryEntry<G>>,
    indicator_bar: Arc<Box<ProgressBar>>,
) where
    G: game::Feature + game::SingleWinner + Clone + Send + Sync + 'static,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
    GB: GameBuilder<G>,
{
    let puct = PUCT::<G> {
        config: puct_settings,
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        evaluate: Arc::new(move |pov: G::Player, board: &G| {
            super::evaluator::prediction(prediction_channel.clone(), pov, board.clone(), false)
        }),
        _g: PhantomData,
    };

    // Generate games indefinitely.
    loop {
        let mut p1 = puct.create(G::players()[0]);
        let mut p2 = puct.create(G::players()[1]);

        let mut state: G =
            game_builder.create(*G::players().choose(&mut rand::thread_rng()).unwrap());

        let mut history_state = vec![];
        let mut history_policy = vec![];
        let mut history_value = vec![];
        let mut history_action = vec![];
        let mut history_reward = vec![];
        let mut history_turn = vec![];

        while { !state.is_finished() } {
            let policy = if state.turn() == G::players()[0] {
                &mut p1
            } else {
                &mut p2
            };
            let action = policy.play(&state);

            /* Save search statistics */
            let game_node = policy.root.as_ref().unwrap();
            let visit_count = game_node.borrow().info.node.count;

            let monte_carlo_distribution: HashMap<G::Move, f32> = HashMap::from_iter(
                game_node
                    .borrow()
                    .info
                    .moves
                    .iter()
                    .map(|(k, v)| (*k, v.N_a / visit_count)),
            );

            let root_value: f32 = game_node
                .borrow()
                .info
                .moves
                .iter()
                .map(|(_, v)| v.reward + (puct_settings.DISCOUNT * v.Q * v.N_a / visit_count))
                .sum();

            history_turn.push(state.turn().into() as f32);
            history_state.push(state.state_to_feature(state.turn()).insert_axis(Axis(0)));
            history_policy
                .push(G::moves_to_feature(&monte_carlo_distribution).insert_axis(Axis(0)));
            history_value.push(Array::from_elem(ndarray::Ix1(1), root_value));
            history_action.push(G::move_to_feature(action).insert_axis(Axis(0)));

            let reward = state.play(&action);
            history_reward.push(Array::from_elem(ndarray::Ix1(1), reward));
        }

        let history_state_view: Vec<_> = history_state.iter().map(|x| x.view()).collect();
        let history_policy_view: Vec<_> = history_policy.iter().map(|x| x.view()).collect();
        let history_action_view: Vec<_> = history_action.iter().map(|x| x.view()).collect();
        let history_value_view: Vec<_> = history_value.iter().map(|x| x.view()).collect();
        let history_reward_view: Vec<_> = history_reward.iter().map(|x| x.view()).collect();

        output_chan
            .send(GameHistoryEntry {
                state: ndarray::stack(Axis(0), &history_state_view).unwrap(),
                policy: ndarray::stack(Axis(0), &history_policy_view).unwrap(),
                action: ndarray::stack(Axis(0), &history_action_view).unwrap(),
                value: ndarray::stack(Axis(0), &history_value_view).unwrap(),
                reward: ndarray::stack(Axis(0), &history_reward_view).unwrap(),
                turn: history_turn,
            })
            .ok()
            .unwrap();

        indicator_bar.inc(1 as u64);
    }
}

/**
 *  AlphaZero self-play games generator
 *
 *  Spawn several tasks (number according to settings) that performs self-play games
 *  using the AlphaZero policy, sending them in the `output_chan` channel.
 *
 *  # Params
 *
 *  - `puct_settings`: configuration for PUCT policy.
 *  - `game_builder`: game builder.
 *  - `prediction_tensorflow`: interface for the prediction network.
 *  - `output_chan`: communication channel to emit the generated games.
 *
 *  # Panics
 *
 *  This function will panic if the evaluator shapes doesn't fit,
 *  or if the CUDA executor goes out of memory.
 */
pub fn alphazero_game_generator<G, GB>(
    puct_settings: PUCTSettings,
    game_builder: GB,
    prediction_tensorflow: Arc<(AtomicBool, RwLock<(Graph, Session)>)>,
    output_chan: mpsc::SyncSender<GameHistoryEntry<G>>,
) where
    G: game::Feature + game::SingleWinner + Send + Sync + Clone + Hash + Eq + 'static,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
    GB: GameBuilder<G> + Copy + Sync + Send + 'static,
{
    let indicator_bar = ProgressBar::new_spinner();
    indicator_bar.set_style(
        ProgressStyle::default_spinner()
            .template("[{spinner}] {wide_bar} {pos} games generated ({elapsed_precise})"),
    );
    indicator_bar.enable_steady_tick(200);
    let bar_box = Arc::new(Box::new(indicator_bar));

    let mut join_handles = vec![];
    let mut join_handles_ev = vec![];

    let state: G = game_builder.create(G::players()[0]);

    let board_size = state.state_dimension().size();
    let action_size = G::action_dimension().size();

    log::debug!("Action: {}", action_size);
    log::debug!("Board: {}", board_size);

    for _ in 0..settings::GPU_N_EVALUATORS {
        let (pred_tx, pred_rx) =
            mpsc::sync_channel::<PredictionEvaluatorChannel>(2 * settings::GPU_BATCH_SIZE);

        for _ in 0..settings::GPU_N_GENERATORS {
            let pred_tx = pred_tx.clone();
            let output_tx = output_chan.clone();
            let czop = bar_box.clone();

            join_handles.push(thread::spawn(move || {
                alphazero_game_generator_task(puct_settings, game_builder, pred_tx, output_tx, czop)
            }));
        }

        let prediction_tensorflow = prediction_tensorflow.clone();

        join_handles_ev.push(thread::spawn(move || {
            super::evaluator::prediction_task(
                board_size,
                action_size,
                1,
                prediction_tensorflow.as_ref(),
                pred_rx,
            )
        }));
    }

    for join_handle in join_handles.drain(..) {
        join_handle.join().unwrap();
    }

    for join_handle in join_handles_ev.drain(..) {
        join_handle.join().unwrap();
    }
}
