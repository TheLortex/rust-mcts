pub mod policies;

use std::collections::HashMap;
use std::iter::FromIterator;

const MODEL_PATH: &str = "models/breakthrough";

use tokio::prelude::*;
use tokio::sync::{mpsc, oneshot};


const BATCH_SIZE: usize = 64;
const N_EVALUATORS: usize = 7;
const N_GENERATORS: usize = 3 * BATCH_SIZE / 2;

/*
const BATCH_SIZE: usize = 1;
const N_EVALUATORS: usize = 1;
const N_GENERATORS: usize = 1;
*/

use self::policies::{puct, puct::PUCT, AsyncMultiplayerPolicy, AsyncMultiplayerPolicyBuilder};
use super::game::breakthrough::{Breakthrough, BreakthroughBuilder, Color, Move, K};
use super::game;
use super::game::{BaseGame, Feature, MultiplayerGame, MultiplayerGameBuilder};
use std::marker::PhantomData;

use indicatif::{ProgressBar, ProgressStyle};

use tensorflow::{Graph, Session, SessionRunArgs, Tensor};

use super::misc::tensorflow_call;

type EvaluatorChannel = (Tensor<f32>, oneshot::Sender<(Tensor<f32>, Tensor<f32>)>);
type GameHistoryChannel = (Color, Vec<(Breakthrough, HashMap<Move, f32>)>);

use ndarray::{Array, ArrayBase, Dimension};

async fn breakthrough_evaluator_batch<'a>(
    mut sender: mpsc::Sender<EvaluatorChannel>,
    pov: Color,
    board_history: Vec<Breakthrough>,
) -> (Array<f32, <Breakthrough as Feature>::ActionDim>, f32) {
    let input_dimensions = Breakthrough::state_dimension();
    let input_stride = input_dimensions.size();
    // add one dimension to the tensor, this looks bad.
    let w_history_dim: Vec<u64> = [
        &[board_history.len()],
        input_dimensions.as_array_view().to_slice().unwrap(),
    ]
    .concat()
    .iter()
    .map(|x| *x as u64)
    .collect();
    let mut board_tensor = Tensor::new(&w_history_dim);
    for (i, board) in board_history.iter().enumerate() {
        board_tensor[i * input_stride..(i + 1) * input_stride]
            .clone_from_slice(&board.state_to_feature(pov).into_raw_vec());
    }
    

    let (resp_tx, resp_rx) = oneshot::channel();
    sender.send((board_tensor, resp_tx)).await.ok().unwrap();
    let (policy_tensor, value_tensor) = resp_rx.await.unwrap();

    let policy =
        ArrayBase::from_shape_vec(Breakthrough::action_dimension(), (&policy_tensor).to_vec())
            .unwrap();
    let value = value_tensor[0];
    (policy, value)
}

async fn game_generator_task(
    sender: mpsc::Sender<EvaluatorChannel>,
    mut output_chan: mpsc::Sender<GameHistoryChannel>,
    bar: Arc<Box<ProgressBar>>,
) {
    let gb = BreakthroughBuilder {};

    let puct = PUCT {
        C_PUCT: 4.,
        N_HISTORY: 2,
        evaluate: &|pov: Color, board_history: &[Breakthrough]| {
            let bidule: Vec<Breakthrough> = board_history.iter().map(|x| (*x).clone()).collect();
            breakthrough_evaluator_batch(sender.clone(), pov, bidule)
        }, 
        _g: PhantomData,
    };

    loop {
        let mut p1 = puct.create(Color::Black);
        let mut p2 = puct.create(Color::White);

        let mut state_history: Vec<Breakthrough>  = vec![gb.create(Color::random())];
        let mut history = vec![];

        while { state_history.last().unwrap().winner().is_none() } {
            let policy = if state_history.last().unwrap().turn() == Color::Black {
                &mut p1
            } else {
                &mut p2
            };
            let action = policy.play(&state_history).await;

            let mut game = state_history.last().unwrap().clone();
            let game_node = policy.inner.tree.get(&game.hash()).unwrap();
            let monte_carlo_distribution: HashMap<Move, f32> = HashMap::from_iter(
                game_node
                    .moves
                    .iter()
                    .map(|(k, v)| (*k, v.N_a / game_node.count)),
            );
            history.push((game.clone(), monte_carlo_distribution));

            game.play(&action);
            state_history.push(game);
        }

        output_chan
            .send((state_history.last().unwrap().winner().unwrap(), history))
            .await
            .ok()
            .unwrap();
        bar.inc(1 as u64);
    }
}

async fn game_evaluator_task<G: Feature>(
    g_and_s: Arc<RwLock<(Graph, Session)>>,
    mut receiver: mpsc::Receiver<EvaluatorChannel>,
    bar: Arc<Box<ProgressBar>>,
) {
    println!("Starting game evaluator..");

    let feature_size: usize = G::state_dimension().size() * 2; //TODO: beurk
    let policy_size: usize = G::action_dimension().size();
    let mut input_tensor: Tensor<f32> = Tensor::new(&[BATCH_SIZE as u64, feature_size as u64]);
    let mut tx_buf = vec![];
    let mut idx = 0;

    while let Some((input, tx)) = receiver.recv().await {
        input_tensor[idx * feature_size..(idx + 1) * feature_size].clone_from_slice(&input);
        tx_buf.push(tx);
        idx += 1;

        if idx == BATCH_SIZE {
            //bar.inc(BATCH_SIZE as u64);
            idx = 0;
            let (policies, values) = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tensorflow_call(&session, &graph, &input_tensor)
            };
            for i in (0..BATCH_SIZE).rev() {
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

pub async fn game_generator(
    graph_and_session: Arc<RwLock<(Graph, Session)>>,
    output_chan: mpsc::Sender<GameHistoryChannel>,
) {
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

    for _ in 0..N_EVALUATORS {
        let (tx, rx) = mpsc::channel::<EvaluatorChannel>(2 * BATCH_SIZE);
        for _ in 0..N_GENERATORS {
            let cmd_tx = tx.clone();
            let output_tx = output_chan.clone();
            let czop = bar_box.clone();
            join_handles.push(tokio::spawn(async move {
                game_generator_task(cmd_tx, output_tx, czop).await
            }));
        }
        drop(tx);
        let czop2 = bar_box2.clone();
        let g_and_s = graph_and_session.clone();
        join_handles_ev.push(tokio::spawn(async move {
            game_evaluator_task::<Breakthrough>(g_and_s, rx, czop2).await
        }));
    }

    for join_handle in join_handles.drain(..) {
        join_handle.await.unwrap();
    }

    for join_handle in join_handles_ev.drain(..) {
        join_handle.await.unwrap();
    }
}
