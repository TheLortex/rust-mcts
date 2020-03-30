use crate::game;

use crate::misc::tf;
use crate::settings;

use ndarray::{Array, ArrayBase, Dimension};
use tensorflow::{Graph, Session, Tensor};
use std::sync::mpsc;
use gstuff::oneshot;

use crate::game::meta::simulated::DynamicsNetworkOutput;

const WARN_ON_GPU_UNDERUSAGE: bool = false;

/*
 * EvaluatorChannels takes a tensor and a way to send back the inference result.
 */
pub type PredictionEvaluatorChannel = (Tensor<f32>, oneshot::Sender<(Tensor<f32>, Tensor<f32>)>);
pub type RepresentationEvaluatorChannel = (Tensor<f32>, oneshot::Sender<Tensor<f32>>);
pub type DynamicsEvaluatorChannel = ((Tensor<f32>, Tensor<f32>), oneshot::Sender<(Tensor<f32>, Tensor<f32>)>);

pub fn prediction<G>(
    sender: mpsc::SyncSender<PredictionEvaluatorChannel>,
    pov: G::Player,
    board: G,
) -> (Array<f32, G::ActionDim>, f32)
where
    G: game::Feature,
{
    let board_features = board.state_to_feature(pov);
    let board_tensor = Tensor::new(
        &board_features.shape()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&board_features.into_raw_vec())
    .unwrap();

    let (resp_tx, resp_rx) = oneshot::oneshot();

    sender.send((board_tensor, resp_tx)).ok().unwrap();
    let (policy_tensor, value_tensor) = resp_rx.recv().unwrap();

    let policy =
        ArrayBase::from_shape_vec(G::action_dimension(), (&policy_tensor).to_vec()).unwrap();
    let value = value_tensor[0];
    (policy, value)
}

pub fn representation<G,H>(
    sender: mpsc::SyncSender<RepresentationEvaluatorChannel>,
    hidden_shape: H,
    state: Array<f32, G>,
) -> Array<f32, H>
where
    G: Dimension,
    H: Dimension
{
    let board_tensor = Tensor::new(
        &state.shape()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&state.into_raw_vec())
    .unwrap();

    let (resp_tx, resp_rx) = oneshot::oneshot();

    sender.send((board_tensor, resp_tx)).ok().unwrap();
    let repr_board_tensor = resp_rx.recv().unwrap();
    

    ArrayBase::from_shape_vec(hidden_shape, (&repr_board_tensor).to_vec()).unwrap()
}

pub fn dynamics<G,H>(
    sender: mpsc::SyncSender<DynamicsEvaluatorChannel>,
    hidden_shape: H,
    board: Array<f32, H>,
    action: Array<f32, G>
) -> DynamicsNetworkOutput<H>
where
    G: Dimension,
    H: Dimension
{
    let board_tensor = Tensor::new(
        &board.shape()
        .iter()
        .map(|i| *i as u64)
        .collect::<Vec<u64>>(),
    )
    .with_values(&board.into_raw_vec())
    .unwrap();

    let action_tensor = Tensor::new(
        &action.shape()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&action.into_raw_vec())
    .unwrap();

    let (resp_tx, resp_rx) = oneshot::oneshot();

    sender.send(((board_tensor, action_tensor), resp_tx)).ok().unwrap();
    let (next_board_tensor, reward) = resp_rx.recv().unwrap();

    let tensor_as_vec = (&next_board_tensor).to_vec();
    let hidden_state = ArrayBase::from_shape_vec(hidden_shape, tensor_as_vec).unwrap();
    DynamicsNetworkOutput {
        reward: reward[0],
        hidden_state,
    }
}

use std::time::{Instant, Duration};
use std::sync::{Arc, RwLock};

pub fn prediction_task(
    repr_size: usize,
    action_size: usize,
    g_and_s: Arc<RwLock<(Graph, Session)>>,
    receiver: mpsc::Receiver<PredictionEvaluatorChannel>,
) {
    println!("Starting prediction evaluator..");

    
    let mut repr_tensor: Tensor<f32> =
        Tensor::new(&[settings::GPU_BATCH_SIZE as u64, repr_size as u64]);
    let mut tx_buf = vec![];
    let mut idx = 0;

    let mut last_time = Instant::now();
    let timeout =  Duration::from_nanos(1_000_000_000/10_000); //1kHz: Should be the number of CPU-GPU roundtrip/sec.

    let mut last_warning = Instant::now();
    let last_warning_duration = Duration::from_secs(10);

    loop {
        let recv_result = receiver.recv_deadline(last_time + timeout);

        let send_batch = match recv_result {
            Ok((repr, tx)) => {
                repr_tensor[idx * repr_size..(idx + 1) * repr_size].clone_from_slice(&repr);
                tx_buf.push(tx);
                idx += 1;
                idx == settings::GPU_BATCH_SIZE
            },
            Err(mpsc::RecvTimeoutError::Timeout) => idx > 0,
            x => panic!(x)
        };


        if send_batch {
            if WARN_ON_GPU_UNDERUSAGE && idx < settings::GPU_BATCH_SIZE/2 && (Instant::now() - last_warning) > last_warning_duration {
                last_warning = Instant::now();
                log::warn!("Prediction: GPU underused.");
                log::warn!("Reduce batch size or increase workers. ({}%)", 100*idx/settings::GPU_BATCH_SIZE);
                log::warn!("");
            }

            let (policies, values) = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tf::call_prediction(&session, &graph, &repr_tensor)
            };

            for i in (0..idx).rev() {
                let policy = Tensor::from(&policies[i * action_size..(i + 1) * action_size]);
                let value = Tensor::from(values[i]);
                tx_buf.pop().unwrap().send((policy, value));
            }
            idx = 0;
            tx_buf.clear();
            last_time = Instant::now();
        }
    }
}


pub fn dynamics_task(
    repr_size: usize,
    action_size: usize,
    g_and_s: Arc<RwLock<(Graph, Session)>>,
    receiver: mpsc::Receiver<DynamicsEvaluatorChannel>,
) {
    println!("Starting dynamics evaluator..");

    let mut repr_tensor: Tensor<f32> =
        Tensor::new(&[settings::GPU_BATCH_SIZE as u64, repr_size as u64]);

    let mut action_tensor: Tensor<f32> =
        Tensor::new(&[settings::GPU_BATCH_SIZE as u64, action_size as u64]);


    let mut tx_buf = vec![];
    let mut idx = 0;

    let mut last_time = Instant::now();
    let timeout =  Duration::from_nanos(1_000_000_000/10_000); //1kHz: Should be the number of CPU-GPU roundtrip/sec.

    let mut last_warning = Instant::now();
    let last_warning_duration = Duration::from_secs(10);

    loop {
        let recv_result = receiver.recv_deadline(last_time + timeout);

        let send_batch = match recv_result {
            Ok(((repr, action), tx)) => {
                repr_tensor[idx * repr_size..(idx + 1) * repr_size].clone_from_slice(&repr);
                action_tensor[idx * action_size..(idx + 1) * action_size].clone_from_slice(&action);
                tx_buf.push(tx);
                idx += 1;
                idx == settings::GPU_BATCH_SIZE
            },
            Err(mpsc::RecvTimeoutError::Timeout) => idx > 0,
            x => panic!(x)
        };


        if send_batch {
            if WARN_ON_GPU_UNDERUSAGE && idx < settings::GPU_BATCH_SIZE/2 && (Instant::now() - last_warning) > last_warning_duration {
                last_warning = Instant::now();
                log::warn!("Prediction: GPU underused.");
                log::warn!("Reduce batch size or increase workers. ({}%)", 100*idx/settings::GPU_BATCH_SIZE);
                log::warn!("");
            }

            let (rewards, next_reprs) = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tf::call_dynamics(&session, &graph, &repr_tensor, &action_tensor)
            };

            for i in (0..idx).rev() {
                let next_repr = Tensor::from(&next_reprs[i * repr_size..(i + 1) * repr_size]);
                let reward = Tensor::from(rewards[i]);
                tx_buf.pop().unwrap().send((next_repr, reward));
            }
            idx = 0;
            tx_buf.clear();
            last_time = Instant::now();
        }
    }
}


pub fn representation_task(
    board_size: usize,
    repr_size: usize,
    g_and_s: Arc<RwLock<(Graph, Session)>>,
    receiver: mpsc::Receiver<RepresentationEvaluatorChannel>,
) {
    println!("Starting representation evaluator..");

    
    let mut board_tensor: Tensor<f32> =
        Tensor::new(&[settings::GPU_BATCH_SIZE as u64, board_size as u64]);
    let mut tx_buf = vec![];
    let mut idx = 0;

    let mut last_time = Instant::now();
    let timeout =  Duration::from_nanos(1_000_000_000/10_000); //10kHz: Should be the number of CPU-GPU roundtrip/sec.

    let mut last_warning = Instant::now();
    let last_warning_duration = Duration::from_secs(10);

    loop {
        let recv_result = receiver.recv_deadline(last_time + timeout);

        let send_batch = match recv_result {
            Ok((board, tx)) => {
                board_tensor[idx * board_size..(idx + 1) * board_size].clone_from_slice(&board);
                tx_buf.push(tx);
                idx += 1;
                idx == settings::GPU_BATCH_SIZE
            },
            Err(mpsc::RecvTimeoutError::Timeout) => idx > 0,
            x => panic!(x)
        };


        if send_batch {
            if WARN_ON_GPU_UNDERUSAGE && idx < settings::GPU_BATCH_SIZE/2 && (Instant::now() - last_warning) > last_warning_duration {
                last_warning = Instant::now();
                log::warn!("Prediction: GPU underused.");
                log::warn!("Reduce batch size or increase workers. ({}%)", 100*idx/settings::GPU_BATCH_SIZE);
                log::warn!("");
            }

            let reprs = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tf::call_representation(&session, &graph, &board_tensor)
            };


            for i in (0..idx).rev() {
                let repr = Tensor::from(&reprs[i * repr_size..(i + 1) * repr_size]);
                tx_buf.pop().unwrap().send(repr);
            }
            idx = 0;
            tx_buf.clear();
            last_time = Instant::now();
        }
    }
}
