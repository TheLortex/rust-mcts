use super::game::breakthrough::{Breakthrough, Move, K};
use super::game::BaseGame;

use std::collections::HashMap;
use std::iter::FromIterator;

use tensorflow::{Graph, Session, SessionRunArgs, Tensor};
use tensorflow::{Code, SessionOptions, Status};

pub fn tensorflow_call(
    session: &Session,
    graph: &Graph,
    input: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    let board_op = graph
        .operation_by_name_required("serving_default_board")
        .unwrap();
    let output_op = graph
        .operation_by_name_required("StatefulPartitionedCall")
        .unwrap();
    let mut args = SessionRunArgs::new();
    args.add_feed(&board_op, 0, input);

    let policy_req = args.request_fetch(&output_op, 0);
    let value_req = args.request_fetch(&output_op, 1);
    session.run(&mut args).unwrap();

    let policy_tensor: Tensor<f32> = args.fetch(policy_req).unwrap();
    let value_tensor: Tensor<f32> = args.fetch(value_req).unwrap();
    (policy_tensor, value_tensor)
}

pub fn breakthrough_evaluator(
    session: &Session,
    graph: &Graph,
    board: &Breakthrough,
) -> (HashMap<Move, f32>, f32) {
    let K_ = K as u64;
    let mut board_tensor: Tensor<f32> = Tensor::new(&[1, 2 * K_ * K_ + 1]);
    for (j, item) in board.serialize().iter().enumerate() {
        board_tensor[j] = *item;
    }

    let (policy_tensor, value_tensor) = tensorflow_call(session, graph, &board_tensor);

    let policy = HashMap::from_iter(
        board
            .possible_moves()
            .into_iter()
            .map(|m| (*m, policy_tensor[m.serialize()])),
    );
    let value = value_tensor[0];
    (policy, value)
}

async fn breakthrough_evaluator_batch(
    mut sender: mpsc::Sender<EvaluatorChannel>,
    board: Breakthrough,
) -> (HashMap<Move, f32>, f32) {
    let K_ = K as u64;
    let mut board_tensor: Tensor<f32> = Tensor::new(&[1, 2 * K_ * K_ + 1]);
    for (j, item) in board.serialize().iter().enumerate() {
        board_tensor[j] = *item;
    }

    let (resp_tx, resp_rx) = oneshot::channel();
    sender.send((board_tensor, resp_tx)).await.ok().unwrap();
    let (policy_tensor, value_tensor) = resp_rx.await.unwrap();

    let policy = HashMap::from_iter(
        board
            .possible_moves()
            .into_iter()
            .map(|m| (*m, policy_tensor[m.serialize()])),
    );
    let value = value_tensor[0];
    (policy, value)
}

const MODEL_PATH: &str = "models/breakthrough";

use tokio::prelude::*;

use tokio::sync::{mpsc, oneshot};

const BATCH_SIZE: usize = 5000;
const BATCH_TIMEOUT: usize = BATCH_SIZE * 1000 * 1000 * 1000 / (10 * 1000 * 1000); // in nanos

use super::game::breakthrough::{BreakthroughBuilder, Color};
use super::game::{MultiplayerGame, MultiplayerGameBuilder};
use super::policies::{puct_async::PUCT, AsyncMultiplayerPolicyBuilder, AsyncMultiplayerPolicy};
use std::marker::PhantomData;

async fn game_generator_task(sender: mpsc::Sender<EvaluatorChannel>) {
    println!("Starting game generator..");
    let gb = BreakthroughBuilder {};

    let puct = PUCT {
        C_PUCT: 4.,
        evaluate: &|board: &Breakthrough| breakthrough_evaluator_batch(sender.clone(), board.clone()), // hacky ? should not have to clone..
        _g: PhantomData,
    };

    loop {
        let mut p1 = puct.create(Color::Black);
        let mut p2 = puct.create(Color::White);

        let mut game: Breakthrough = gb.create(Color::random());
        let mut history = vec![];

        while { game.winner().is_none() } {
            let policy = if game.turn() == Color::Black { 
                &mut p1
            } else {
                &mut p2
            };
            let action = policy.play(&game).await;
            let game_node = policy.inner.tree.get(&game.hash()).unwrap();
            let monte_carlo_distribution: HashMap<Move, f32> = HashMap::from_iter(
                game_node
                    .moves
                    .iter()
                    .map(|(k, v)| (*k, v.N_a / game_node.count)),
            );
            history.push((game.clone(), monte_carlo_distribution));

            game.play(&action);
        }
    }
}

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

async fn game_evaluator_task(g_and_s: Arc<RwLock<(Graph, Session)>>, bar: Arc<Box<ProgressBar>>, mut receiver: mpsc::Receiver<EvaluatorChannel>) {
    println!("Starting game evaluator..");


    let feature_size: usize = 2 * K * K + 1;
    let policy_size: usize = 3 * K * K;
    let mut input_tensor: Tensor<f32> = Tensor::new(&[BATCH_SIZE as u64, feature_size as u64]);
    let mut tx_buf = vec![];
    let mut idx = 0;

    while let Some((input, tx)) = receiver.recv().await {
    
        input_tensor[idx*feature_size..(idx+1)*feature_size].clone_from_slice(&input);
        tx_buf.push(tx);
        idx += 1;

        if idx == BATCH_SIZE {
            bar.inc(BATCH_SIZE as u64);
            idx = 0;
                
            let (policies, values) = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tensorflow_call(&session, &graph, &input_tensor)
            };
            for i in (0..BATCH_SIZE).rev() {
                let policy = Tensor::from(&policies[i*policy_size..(i+1)*policy_size]);
                let value  = Tensor::from(values[i]);
                
                tx_buf.pop().unwrap().send((policy, value)).unwrap();
            }
            tx_buf.clear();
        }
    }
}

type EvaluatorChannel = (Tensor<f32>, oneshot::Sender<(Tensor<f32>, Tensor<f32>)>);

use std::sync::Arc;
use std::sync::RwLock;

pub async fn game_generator() {
    let bar = ProgressBar::new_spinner();
    bar.set_style(ProgressStyle::default_spinner().template("[{spinner}] {wide_bar} {pos} moves generated ({elapsed_precise})"));
    bar.enable_steady_tick(200);
    let bar_box = Arc::new(Box::new(bar));

    let mut join_handles = vec![];
    let mut join_handles_ev = vec![];

    let mut graph = Graph::new();
    let session = {
        let mut options = SessionOptions::new();
        /* To get configuration, use python:
        *      config = tf.ConfigProto()
        *      config.gpu_options.allow_growth = True
        *      config.SerializeToString()
        */
        let configuration_buf = [50, 2, 32, 1];
        options.set_config(&configuration_buf).unwrap(); 
        
        Session::from_saved_model(
            &options,
            &["serve"],
            &mut graph,
            MODEL_PATH,
        ).unwrap()
    };

    let graph_and_session = Arc::new(RwLock::new((graph,session)));

    for _ in 0..8 {
        let (tx, rx) = mpsc::channel::<EvaluatorChannel>(BATCH_SIZE);
        for _ in 0..8*BATCH_SIZE {
            let cmd_tx = tx.clone();
    
            join_handles.push(tokio::spawn(async move {
                game_generator_task(cmd_tx).await
            }));
        }
    
        drop(tx);
        
        let g_and_s = graph_and_session.clone();
        let czop = bar_box.clone();
        join_handles_ev.push(tokio::spawn(async move {
            game_evaluator_task(g_and_s, czop, rx).await
        }));
    }

    for join_handle in join_handles.drain(..) {
        join_handle.await.unwrap(); // ?????????
    }

    for join_handle in join_handles_ev.drain(..) {
        join_handle.await.unwrap(); // ?????????
    }
}
