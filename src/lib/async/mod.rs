use crate::game;

use crate::game::{BaseGame, Feature, MultiplayerGame, MultiplayerGameBuilder};
use crate::misc::tensorflow_call;
use crate::policies::{
    mcts::puct::{BatchedPUCT, PUCTSettings},
    AsyncMultiplayerPolicy, AsyncMultiplayerPolicyBuilder,
};
use crate::settings;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, ArrayBase, Dimension};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::marker::PhantomData;
use tensorflow::{Graph, Session, Tensor};
use tokio::sync::{mpsc, oneshot};

type EvaluatorChannel = (Tensor<f32>, oneshot::Sender<(Tensor<f32>, Tensor<f32>)>);
pub type GameHistoryChannel<G> = (
    <G as MultiplayerGame>::Player,
    Vec<(G, HashMap<<G as BaseGame>::Move, f32>)>,
);

async fn breakthrough_evaluator_batch<'a, G>(
    mut sender: mpsc::Sender<EvaluatorChannel>,
    pov: G::Player,
    board: G,
) -> (Array<f32, G::ActionDim>, f32)
where
    G: game::Feature,
{
    let input_dimensions = G::state_dimension();
    let board_tensor = Tensor::new(
        &input_dimensions
            .as_array_view()
            .to_slice()
            .unwrap()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&board.state_to_feature(pov).into_raw_vec())
    .unwrap();

    let (resp_tx, resp_rx) = oneshot::channel();
    sender.send((board_tensor, resp_tx)).await.ok().unwrap();
    let (policy_tensor, value_tensor) = resp_rx.await.unwrap();

    let policy =
        ArrayBase::from_shape_vec(G::action_dimension(), (&policy_tensor).to_vec()).unwrap();
    let value = value_tensor[0];
    (policy, value)
}

async fn game_generator_task<G, GB>(
    puct_settings: PUCTSettings,
    game_builder: GB,
    sender: mpsc::Sender<EvaluatorChannel>,
    mut output_chan: mpsc::Sender<GameHistoryChannel<G>>,
    bar: Arc<Box<ProgressBar>>,
) where
    G: game::Feature + Send + Sync + Clone,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
    GB: MultiplayerGameBuilder<G>,
{
    let puct = BatchedPUCT {
        s: puct_settings,
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        evaluate: |pov: G::Player, board: &G| {
            breakthrough_evaluator_batch(sender.clone(), pov, board.clone())
        },
        _g: PhantomData,
    };

    loop {
        let mut p1 = puct.create(G::players()[0]);
        let mut p2 = puct.create(G::players()[1]);

        let mut state: G =
            game_builder.create(*G::players().choose(&mut rand::thread_rng()).unwrap());
        let mut history = vec![];

        while { !state.is_finished() } {
            let policy = if state.turn() == G::players()[0] {
                &mut p1
            } else {
                &mut p2
            };
            let action = policy.play(&state).await;

            let game_node = policy.inner.b.tree.get(&state.hash()).unwrap();
            let monte_carlo_distribution: HashMap<G::Move, f32> = HashMap::from_iter(
                game_node
                    .moves
                    .iter()
                    .map(|(k, v)| (*k, v.N_a / game_node.count)),
            );
            history.push((state.clone(), monte_carlo_distribution));

            state.play(&action);
        }

        // warning: this assumes a 2-player game.
        let winner = if state.has_won(G::players()[0]) {
            G::players()[0]
        } else {
            G::players()[1]
        };

        output_chan.send((winner, history)).await.ok().unwrap();

        bar.inc(1 as u64);
    }
}

async fn game_evaluator_task<G: Feature>(
    g_and_s: Arc<RwLock<(Graph, Session)>>,
    mut receiver: mpsc::Receiver<EvaluatorChannel>,
    _bar: Arc<Box<ProgressBar>>,
) {
    println!("Starting game evaluator..");

    let feature_size: usize = G::state_dimension().size();
    let policy_size: usize = G::action_dimension().size();
    let mut input_tensor: Tensor<f32> =
        Tensor::new(&[settings::GPU_BATCH_SIZE as u64, feature_size as u64]);
    let mut tx_buf = vec![];
    let mut idx = 0;

    while let Some((input, tx)) = receiver.recv().await {
        input_tensor[idx * feature_size..(idx + 1) * feature_size].clone_from_slice(&input);
        tx_buf.push(tx);
        idx += 1;

        if idx == settings::GPU_BATCH_SIZE {
            //bar.inc(BATCH_SIZE as u64);
            idx = 0;
            let (policies, values) = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tensorflow_call(&session, &graph, &input_tensor)
            };
            for i in (0..settings::GPU_BATCH_SIZE).rev() {
                let policy = Tensor::from(&policies[i * policy_size..(i + 1) * policy_size]);
                let value = Tensor::from(values[i]);
                tx_buf.pop().unwrap().send((policy, value)).unwrap();
            }
            tx_buf.clear();
        }
    }
}

use std::sync::Arc;
use std::sync::RwLock;

pub async fn game_generator<G, GB>(
    puct_settings: PUCTSettings,
    game_builder: GB,
    graph_and_session: Arc<RwLock<(Graph, Session)>>,
    output_chan: mpsc::Sender<GameHistoryChannel<G>>,
) where
    G: game::Feature + Send + Sync + Clone + 'static,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
    GB: MultiplayerGameBuilder<G> + Copy + Send + 'static,
{
    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::default_spinner()
            .template("[{spinner}] {wide_bar} {pos} games generated ({elapsed_precise})"),
    );
    bar.enable_steady_tick(200);
    let bar_box = Arc::new(Box::new(bar));

    let bar2 = ProgressBar::new_spinner();
    bar2.set_style(
        ProgressStyle::default_spinner()
            .template("[{spinner}] {wide_bar} {pos} moves generated ({elapsed_precise})"),
    );
    //bar2.enable_steady_tick(200);
    let bar_box2 = Arc::new(Box::new(bar2));

    let mut join_handles = vec![];
    let mut join_handles_ev = vec![];

    //let gb_box = Arc::new(Box::new(game_builder));

    for _ in 0..settings::GPU_N_EVALUATORS {
        let (tx, rx) = mpsc::channel::<EvaluatorChannel>(2 * settings::GPU_BATCH_SIZE);
        for _ in 0..settings::GPU_N_GENERATORS {
            let cmd_tx = tx.clone();
            let output_tx = output_chan.clone();
            let czop = bar_box.clone();

            join_handles.push(tokio::spawn({
                game_generator_task(puct_settings, game_builder, cmd_tx, output_tx, czop)
            }));
        }
        drop(tx);
        let czop2 = bar_box2.clone();
        let g_and_s = graph_and_session.clone();
        join_handles_ev.push(tokio::spawn(async move {
            game_evaluator_task::<G>(g_and_s, rx, czop2).await
        }));
    }

    for join_handle in join_handles.drain(..) {
        join_handle.await.unwrap();
    }

    for join_handle in join_handles_ev.drain(..) {
        join_handle.await.unwrap();
    }
}
