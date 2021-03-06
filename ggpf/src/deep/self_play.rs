
//!
//! # Asynchronous evaluation functions.
//!
//! For each kind of evaluation there is an evaluator that can be
//! provided to PUCT/Muz and an associated evaluator task that sends
//! the request to the GPU while batching them.
//!


use crate::deep::evaluator::PredictionEvaluatorChannel;
use crate::game::GameBuilder;
use crate::game::*;
use crate::policies::mcts::puct::PUCT;
use crate::policies::mcts::{muz, puct};
use crate::policies::{
    mcts::muz::{Muz, MuzPolicy},
    MultiplayerPolicy, MultiplayerPolicyBuilder,
};
use crate::settings;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Axis, Dimension, Ix1};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::sync::Arc;
use tokio::sync::mpsc;

///
/// Game history data generated from self-play
///
pub struct GameHistoryEntry<G>
where
    G: Features,
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
async fn muzero_game_generator_task<GB, B, A>(
    config: muz::MuZeroConfig<B, A>,
    game_builder: GB,
    channels: muz::MuzEvaluatorChannels,
    mut output_chan: mpsc::Sender<GameHistoryEntry<GB::G>>,
    indicator_bar: Arc<Box<ProgressBar>>,
) where
    GB::G: Features + Send + Sync + 'static,
    <GB::G as Base>::Move: Send + Sync,
    <GB::G as Game>::Player: Send + Sync,
    GB: GameBuilder,
    A: Dimension,
    B: Dimension,
{
    let muz = Muz {
        n_playouts: config.n_playouts,
        muz: config.muz,
        channels,
    };

    loop {
        let mut policies: HashMap<<GB::G as Game>::Player, MuzPolicy<GB::G>> = HashMap::from_iter(
            <GB::G as Game>::players()
                .iter()
                .map(|i| (*i, muz.create(*i))),
        );

        let random_player = *GB::G::players().choose(&mut rand::thread_rng()).unwrap();
        let mut state: GB::G = game_builder.create(random_player).await;

        let ft = state.get_features();

        let mut history_state = vec![];
        let mut history_policy = vec![];
        let mut history_value = vec![];
        let mut history_action = vec![];
        let mut history_reward = vec![];
        let mut history_turn = vec![];

        while !state.is_finished() {
            let policy = policies.get_mut(&state.turn()).unwrap();
            let action = policy.play(&state).await;

            /* Save search statistics */
            let mcts = policy.mcts.take().unwrap();
            let game_node = mcts.root.as_ref().unwrap();
            let visit_count = game_node.read().unwrap().info.node.count;

            let monte_carlo_distribution: HashMap<<GB::G as Base>::Move, f32> = HashMap::from_iter(
                game_node
                    .read()
                    .unwrap()
                    .info
                    .moves
                    .iter()
                    .map(|(k, v)| (*k, v.N_a / visit_count)),
            );

            let root_value: f32 = game_node
                .read()
                .unwrap()
                .info
                .moves
                .iter()
                .map(|(_, v)| (v.reward + config.muz.puct.discount * v.Q * v.N_a / visit_count))
                .sum();

            history_turn.push(state.turn().into() as f32);
            history_state.push(state.state_to_feature(state.turn()).insert_axis(Axis(0)));
            history_policy.push(
                <GB::G as Features>::moves_to_feature(&ft, &monte_carlo_distribution)
                    .insert_axis(Axis(0)),
            );
            history_value.push(Array::from_elem(ndarray::Ix1(1), root_value));
            history_action
                .push(<GB::G as Features>::move_to_feature(&ft, action).insert_axis(Axis(0)));

            let reward = state.play(&action).await;
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
            .await
            .ok()
            .unwrap();

        indicator_bar.inc(1 as u64);
    }
}

///
/// MuZero self-play games generator
/// Spawn several tasks (number according to settings) that performs self-play games
/// using the MuZero policy, sending them in the `output_chan` channel.
/// # Params
/// - `puct_settings`: configuration for PUCT policy and virtual state dimension.
/// - `game_builder`: game builder.
/// - `prediction_tensorflow`: interface for the prediction network.
/// - `dynamics_tensorflow`: interface for the dynamics network.
/// - `representation_tensorflow`: interface for the representation network.
/// - `output_chan`: communication channel to emit the generated games.
/// # Panics
/// This function will panic if the evaluator shapes doesn't fit,
/// or if the CUDA executor goes out of memory.
///
pub async fn muzero_game_generator<GB, B, A>(
    config: muz::MuZeroConfig<B, A>,
    config_selfplay: settings::SelfPlay,
    game_builder: GB,
    output_chan: mpsc::Sender<GameHistoryEntry<GB::G>>,
) where
    GB::G: Features + Send + Sync + 'static,
    <GB::G as Base>::Move: Send + Sync,
    <GB::G as Game>::Player: Send + Sync,
    GB: GameBuilder + Clone + Sync + Send + 'static,
    A: Dimension + 'static,
    B: Dimension + 'static,
{
    let indicator_bar = ProgressBar::new_spinner();
    indicator_bar.set_style(
        ProgressStyle::default_spinner()
            .template("[{spinner}] {wide_bar} {pos} games generated ({elapsed_precise})"),
    );
    indicator_bar.enable_steady_tick(200);
    let bar_box = Arc::new(Box::new(indicator_bar));

    let mut muzero_evaluators = muz::MuzEvaluators::new(config.clone(), false);

    for _ in 0..config_selfplay.evaluators {
        muzero_evaluators = muzero_evaluators.clone();

        for _ in 0..config_selfplay.generators {
            tokio::spawn(muzero_game_generator_task(
                config.clone(),
                game_builder.clone(),
                muzero_evaluators.get_channels(),
                output_chan.clone(),
                bar_box.clone(),
            ));
        }
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
async fn alphazero_game_generator_task<GB, A, B>(
    config: puct::AlphaZeroConfig<A, B>,
    game_builder: GB,
    prediction_channel: mpsc::Sender<PredictionEvaluatorChannel>,
    mut output_chan: mpsc::Sender<GameHistoryEntry<GB::G>>,
    indicator_bar: Arc<Box<ProgressBar>>,
) where
    GB::G: Features + Clone + Send + Sync + 'static,
    <GB::G as Base>::Move: Send + Sync,
    <GB::G as Game>::Player: Send + Sync,
    GB: GameBuilder,
    A: Dimension,
    B: Dimension,
{
    let puct = PUCT {
        config: config.puct,
        n_playouts: config.n_playouts,
        prediction_channel,
    };

    // Generate games indefinitely.
    loop {
        let mut p1 = puct.create(<GB::G as Game>::players()[0]);
        let mut p2 = puct.create(<GB::G as Game>::players()[1]);
        let random_player = *<GB::G as Game>::players()
            .choose(&mut rand::thread_rng())
            .unwrap();
        let mut state: GB::G = game_builder.create(random_player).await;

        let ft = state.get_features();

        let mut history_state = vec![];
        let mut history_policy = vec![];
        let mut history_value = vec![];
        let mut history_action = vec![];
        let mut history_reward = vec![];
        let mut history_turn = vec![];

        while !state.is_finished() {
            let policy = if state.turn() == <GB::G as Game>::players()[0] {
                &mut p1
            } else {
                &mut p2
            };
            let action = policy.play(&state).await;

            /* Save search statistics */
            let game_node = policy.root.as_ref().unwrap();
            let visit_count = game_node.read().unwrap().info.node.count;

            let monte_carlo_distribution: HashMap<<GB::G as Base>::Move, f32> = HashMap::from_iter(
                game_node
                    .read()
                    .unwrap()
                    .info
                    .moves
                    .iter()
                    .map(|(k, v)| (*k, v.N_a / visit_count)),
            );

            let root_value: f32 = game_node
                .read()
                .unwrap()
                .info
                .moves
                .iter()
                .map(|(_, v)| ((v.reward + config.puct.discount * v.Q) * v.N_a / visit_count))
                .sum();

            history_turn.push(state.turn().into() as f32);
            history_state.push(state.state_to_feature(state.turn()).insert_axis(Axis(0)));
            history_policy.push(
                <GB::G as Features>::moves_to_feature(&ft, &monte_carlo_distribution)
                    .insert_axis(Axis(0)),
            );
            history_value.push(Array::from_elem(ndarray::Ix1(1), root_value));
            history_action
                .push(<GB::G as Features>::move_to_feature(&ft, action).insert_axis(Axis(0)));

            let reward = state.play(&action).await;
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
            .await
            .ok()
            .unwrap();

        indicator_bar.inc(1 as u64);
    }
}

///
///  AlphaZero self-play games generator
///
///  Spawn several tasks (number according to settings) that performs self-play games
///  using the AlphaZero policy, sending them in the `output_chan` channel.
///
///  # Params
///
///  - `puct_settings`: configuration for PUCT policy.
///  - `game_builder`: game builder.
///  - `prediction_tensorflow`: interface for the prediction network.
///  - `output_chan`: communication channel to emit the generated games.
///
///  # Panics
///
///  This function will panic if the evaluator shapes doesn't fit,
///  or if the CUDA executor goes out of memory.
///
pub async fn alphazero_game_generator<GB, A, B>(
    config: puct::AlphaZeroConfig<A, B>,
    config_selfplay: settings::SelfPlay,
    game_builder: GB,
    output_chan: mpsc::Sender<GameHistoryEntry<GB::G>>,
) where
    GB::G: Features + Clone + Send + Sync + 'static,
    <GB::G as Base>::Move: Send + Sync,
    <GB::G as Game>::Player: Send + Sync,
    GB: GameBuilder + Clone + Sync + Send + 'static,
    A: Dimension + 'static,
    B: Dimension + 'static,
{
    let indicator_bar = ProgressBar::new_spinner();
    indicator_bar.set_style(
        ProgressStyle::default_spinner()
            .template("[{spinner}] {wide_bar} {pos} games generated ({elapsed_precise})"),
    );
    indicator_bar.enable_steady_tick(200);
    let bar_box = Arc::new(Box::new(indicator_bar));

    let mut az = puct::AlphaZeroEvaluators::new(config.clone(), false);

    for _ in 0..config_selfplay.evaluators {
        // spawn new workers.
        az = az.clone();

        for _ in 0..config_selfplay.generators {
            tokio::spawn(alphazero_game_generator_task(
                config.clone(),
                game_builder.clone(),
                az.get_channel(),
                output_chan.clone(),
                bar_box.clone(),
            ));
        }
    }
}
