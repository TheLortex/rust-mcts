use crate::deep::tf;
use crate::game;
use crate::game::meta::simulated::DynamicsNetworkOutput;

use ndarray::Axis;
use ndarray::{Array, ArrayBase, Dimension};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::{atomic::AtomicBool, RwLock};
use std::{thread, time};
use tensorflow::{Graph, Session, Tensor};
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::timeout_at;
use tokio::time::{Duration, Instant};

const WARN_ON_GPU_UNDERUSAGE: bool = false;

/// Takes a tensor and a way to send back the inference result for the prediction network.
pub type PredictionEvaluatorChannel = (Tensor<f32>, oneshot::Sender<(Tensor<f32>, Tensor<f32>)>);
/// Takes a tensor and a way to send back the inference result for the representation network.
pub type RepresentationEvaluatorChannel = (Tensor<f32>, oneshot::Sender<Tensor<f32>>);
/// Takes a tensor and a way to send back the inference result for the dynamics network.
pub type DynamicsEvaluatorChannel = (
    (Tensor<f32>, Tensor<f32>),
    oneshot::Sender<(Tensor<f32>, Tensor<f32>)>,
);

/*      HELPERS          */

fn ndarray_to_tensor<D: Dimension>(arr: &Array<f32, D>) -> Tensor<f32> {
    Tensor::new(&arr.shape().iter().map(|i| *i as u64).collect::<Vec<u64>>())
        .with_values(arr.as_slice().unwrap())
        .unwrap()
}

fn tensor_to_ndarray<D: Dimension>(tensor: Tensor<f32>, shape: D) -> Array<f32, D> {
    ArrayBase::from_shape_vec(shape, tensor.to_vec()).unwrap()
}

/*      EVALUATORS       */

/// Prediction evaluator
pub async fn prediction<G>(
    mut sender: mpsc::Sender<PredictionEvaluatorChannel>,
    pov: G::Player,
    board: &G,
    support_size: usize,
) -> (Array<f32, G::ActionDim>, f32)
where
    G: game::Features,
{
    let board_tensor = ndarray_to_tensor(&board.state_to_feature(pov));
    let (resp_tx, resp_rx) = oneshot::channel();
    sender.send((board_tensor, resp_tx)).await.ok().unwrap();
    let (policy_tensor, value_tensor) = resp_rx.await.unwrap();
    let ft = board.get_features();
    let policy = tensor_to_ndarray(policy_tensor, G::action_dimension(&ft));
    let value = if support_size > 0 {
        tf::support_to_value(&value_tensor, 1, support_size)[0]
    } else {
        value_tensor[0]
    };
    (policy, value)
}

/// Representation evaluator
pub async fn representation<G, H>(
    mut sender: mpsc::Sender<RepresentationEvaluatorChannel>,
    hidden_shape: H,
    state: &Array<f32, G>,
) -> Array<f32, H>
where
    G: Dimension,
    H: Dimension,
{
    let board_tensor = ndarray_to_tensor(state);
    let (resp_tx, resp_rx) = oneshot::channel();

    sender.send((board_tensor, resp_tx)).await.ok().unwrap();
    let repr_board_tensor = resp_rx.await.unwrap();

    tensor_to_ndarray(repr_board_tensor, hidden_shape)
}

/// Dynamics evaluator
pub async fn dynamics<G, H>(
    mut sender: mpsc::Sender<DynamicsEvaluatorChannel>,
    board: &Array<f32, H>,
    action: &Array<f32, G>,
    support_size: usize,
) -> DynamicsNetworkOutput<H>
where
    G: Dimension,
    H: Dimension,
{
    let board_dim = board.raw_dim();
    let board_tensor = ndarray_to_tensor(board);
    let action_tensor = ndarray_to_tensor(action);
    let (resp_tx, resp_rx) = oneshot::channel();

    sender
        .send(((board_tensor, action_tensor), resp_tx))
        .await
        .ok()
        .unwrap();
    let (next_board_tensor, reward) = resp_rx.await.unwrap();

    let repr_state = tensor_to_ndarray(next_board_tensor, board_dim);
    let reward = if support_size > 0 {
        tf::support_to_value(&reward, 1, support_size)[0]
    } else {
        reward[0]
    };
    DynamicsNetworkOutput { reward, repr_state }
}

use indicatif::ProgressBar;

/// Prediction task
pub async fn prediction_task(
    batch_size: usize,
    repr_size: usize,
    action_size: usize,
    support_size: usize,
    tensorflow: Arc<(AtomicBool, RwLock<(Graph, Session)>)>,
    mut receiver: mpsc::Receiver<PredictionEvaluatorChannel>,
    bb: Option<Arc<Box<ProgressBar>>>,
) {
    let (writer_lock, g_and_s) = tensorflow.as_ref();
    log::info!("Starting prediction evaluator..");

    let mut repr_tensor: Tensor<f32> = Tensor::new(&[batch_size as u64, repr_size as u64]);
    let mut tx_buf = vec![];
    let mut idx = 0;

    let mut last_time = Instant::now();
    let timeout = Duration::from_nanos(1_000_000_000 / 10_000);

    let mut last_warning = Instant::now();
    let last_warning_duration = Duration::from_secs(10);

    loop {
        let recv_result = timeout_at(last_time + timeout, receiver.recv()).await;

        let send_batch = match recv_result {
            Ok(Some((repr, tx))) => {
                repr_tensor[idx * repr_size..(idx + 1) * repr_size].clone_from_slice(&repr);
                tx_buf.push(tx);
                idx += 1;
                idx == batch_size
            }
            Err(_) => idx > 0,
            _ => return,
        };
        /*
                let send_batch = match recv_result {
                    Some((repr, tx)) => {
                        repr_tensor[idx * repr_size..(idx + 1) * repr_size].clone_from_slice(&repr);
                        tx_buf.push(tx);
                        idx += 1;
                        idx == batch_size
                    }
                    _ => {
                        log::warn!("Channel closed. Leaving.");
                        return;
                    }
                };
        */
        if send_batch {
            if WARN_ON_GPU_UNDERUSAGE
                && idx < batch_size / 2
                && (Instant::now() - last_warning) > last_warning_duration
            {
                last_warning = Instant::now();
                log::warn!("Prediction: GPU underused.");
                log::warn!(
                    "Reduce batch size or increase workers. ({}%)",
                    100 * idx / batch_size
                );
                log::warn!("");
            }

            while writer_lock.load(Ordering::Relaxed) {
                thread::sleep(time::Duration::from_millis(1));
            }

            let (policies, values) = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tf::call_prediction(&session, &graph, &repr_tensor)
            };

            if let Some(x) = bb.as_ref() {
                x.inc(idx as u64);
            }

            for i in (0..idx).rev() {
                let policy = Tensor::from(&policies[i * action_size..(i + 1) * action_size]);
                let value = Tensor::from(&values[i * support_size..(i + 1) * support_size]);
                tx_buf.pop().unwrap().send((policy, value)).unwrap();
            }
            idx = 0;
            tx_buf.clear();
        }
        last_time = Instant::now();
    }
}

/// Dynamics task
pub async fn dynamics_task(
    batch_size: usize,
    repr_size: usize,
    action_size: usize,
    support_size: usize,
    tensorflow: Arc<(AtomicBool, RwLock<(Graph, Session)>)>,
    mut receiver: mpsc::Receiver<DynamicsEvaluatorChannel>,
) {
    let (writer_lock, g_and_s) = tensorflow.as_ref();
    log::info!("Starting dynamics evaluator..");

    let mut repr_tensor: Tensor<f32> = Tensor::new(&[batch_size as u64, repr_size as u64]);

    let mut action_tensor: Tensor<f32> = Tensor::new(&[batch_size as u64, action_size as u64]);

    let mut tx_buf = vec![];
    let mut idx = 0;

    let mut last_time = Instant::now();
    let timeout = Duration::from_nanos(1_000_000_000 / 10_000); //10kHz: Should be the number of CPU-GPU roundtrip/sec.

    let mut last_warning = Instant::now();
    let last_warning_duration = Duration::from_secs(10);

    loop {
        let recv_result = timeout_at(last_time + timeout, receiver.recv()).await;

        let send_batch = match recv_result {
            Ok(Some(((repr, action), tx))) => {
                repr_tensor[idx * repr_size..(idx + 1) * repr_size].clone_from_slice(&repr);
                action_tensor[idx * action_size..(idx + 1) * action_size].clone_from_slice(&action);
                tx_buf.push(tx);
                idx += 1;
                idx == batch_size
            }
            Err(_) => idx > 0,
            _ => return,
        };
        /*
                let send_batch = match recv_result {
                    Some(((repr, action), tx)) => {
                        repr_tensor[idx * repr_size..(idx + 1) * repr_size].clone_from_slice(&repr);
                        action_tensor[idx * action_size..(idx + 1) * action_size].clone_from_slice(&action);
                        tx_buf.push(tx);
                        idx += 1;
                        idx == batch_size
                    }
                    _ => {
                        log::warn!("Leaving.");
                        return;
                    }
                };
        */
        if send_batch {
            if WARN_ON_GPU_UNDERUSAGE
                && idx < batch_size / 2
                && (Instant::now() - last_warning) > last_warning_duration
            {
                last_warning = Instant::now();
                log::warn!("Prediction: GPU underused.");
                log::warn!(
                    "Reduce batch size or increase workers. ({}%)",
                    100 * idx / batch_size
                );
                log::warn!("");
            }

            while writer_lock.load(Ordering::Relaxed) {
                thread::sleep(time::Duration::from_millis(1));
            }

            let (rewards, next_reprs) = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tf::call_dynamics(&session, &graph, &repr_tensor, &action_tensor)
            };

            for i in (0..idx).rev() {
                let next_repr = Tensor::from(&next_reprs[i * repr_size..(i + 1) * repr_size]);
                let reward = Tensor::from(&rewards[i * support_size..(i + 1) * support_size]);
                tx_buf.pop().unwrap().send((next_repr, reward)).unwrap();
            }
            idx = 0;
            tx_buf.clear();
        }
        last_time = Instant::now();
    }
}

/// Representation task
pub async fn representation_task(
    batch_size: usize,
    board_size: usize,
    repr_size: usize,
    tensorflow: Arc<(AtomicBool, RwLock<(Graph, Session)>)>,
    mut receiver: mpsc::Receiver<RepresentationEvaluatorChannel>,
) {
    let (writer_lock, g_and_s) = tensorflow.as_ref();
    log::info!(
        "Starting representation evaluator.. {}/{}",
        board_size,
        repr_size
    );

    let mut board_tensor: Tensor<f32> = Tensor::new(&[batch_size as u64, board_size as u64]);
    let mut tx_buf = vec![];
    let mut idx = 0;

    let mut last_time = Instant::now();
    let timeout = Duration::from_nanos(1_000_000_000 / 10_000);

    let mut last_warning = Instant::now();
    let last_warning_duration = Duration::from_secs(10);

    loop {
        let recv_result = timeout_at(last_time + timeout, receiver.recv()).await;

        let send_batch = match recv_result {
            Ok(Some((board, tx))) => {
                board_tensor[idx * board_size..(idx + 1) * board_size].clone_from_slice(&board);
                tx_buf.push(tx);
                idx += 1;
                idx == batch_size
            }
            Err(_) => idx > 0,
            _ => return,
        };
        /*
        let send_batch = match recv_result {
            Some((board, tx)) => {
                board_tensor[idx * board_size..(idx + 1) * board_size].clone_from_slice(&board);
                tx_buf.push(tx);
                idx += 1;
                idx == batch_size
            }
            _ => {
                log::warn!("Leaving.");
                return;
            }
        };*/

        if send_batch {
            if WARN_ON_GPU_UNDERUSAGE
                && idx < batch_size / 2
                && (Instant::now() - last_warning) > last_warning_duration
            {
                last_warning = Instant::now();
                log::warn!("Prediction: GPU underused.");
                log::warn!(
                    "Reduce batch size or increase workers. ({}%)",
                    100 * idx / batch_size
                );
                log::warn!("");
            }

            while writer_lock.load(Ordering::Relaxed) {
                thread::sleep(time::Duration::from_millis(1));
            }

            let reprs = {
                let (ref graph, ref session) = *g_and_s.read().unwrap();
                tf::call_representation(&session, &graph, &board_tensor)
            };

            for i in (0..idx).rev() {
                let repr = Tensor::from(&reprs[i * repr_size..(i + 1) * repr_size]);
                tx_buf.pop().unwrap().send(repr).unwrap();
            }
            idx = 0;
            tx_buf.clear();
        }
        last_time = Instant::now();
    }
}

/// Evaluates a game state for PUCT - single batch
pub fn prediction_evaluator_single<G: game::Features>(
    session: &Session,
    graph: &Graph,
    pov: G::Player,
    board: &G,
    support_size: usize,
) -> (Array<f32, G::ActionDim>, f32) {
    let ft = board.get_features();
    let input_dimensions = G::state_dimension(&ft);

    let board_tensor = Tensor::new(
        &input_dimensions
            .insert_axis(Axis(0))
            .as_array_view()
            .to_slice()
            .unwrap()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&board.state_to_feature(pov).into_raw_vec())
    .unwrap();

    let (policy_tensor, value_tensor) = tf::call_prediction(session, graph, &board_tensor);

    let policy = tensor_to_ndarray(policy_tensor, G::action_dimension(&ft));
    let value = if support_size > 0 {
        tf::support_to_value(&value_tensor, 1, support_size)[0]
    } else {
        value_tensor[0]
    };
    (policy, value)
}

/// Dynamics evaluator - single batch
pub fn dynamics_evaluator_single<G: Dimension, H: Dimension>(
    session: &Session,
    graph: &Graph,
    hidden_shape: H,
    board: Array<f32, H>,
    action: Array<f32, G>,
    support_size: usize,
) -> DynamicsNetworkOutput<H> {
    let board_tensor = Tensor::new(
        &board
            .raw_dim()
            .insert_axis(Axis(0))
            .as_array_view()
            .to_slice()
            .unwrap()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&board.into_raw_vec())
    .unwrap();

    let action_tensor = Tensor::new(
        &action
            .raw_dim()
            .insert_axis(Axis(0))
            .as_array_view()
            .to_slice()
            .unwrap()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&action.into_raw_vec())
    .unwrap();

    let (reward, next_board_tensor) =
        tf::call_dynamics(session, graph, &board_tensor, &action_tensor);

    let repr_state = tensor_to_ndarray(next_board_tensor, hidden_shape);
    let reward = if support_size > 0 {
        tf::support_to_value(&reward, 1, support_size)[0]
    } else {
        reward[0]
    };
    DynamicsNetworkOutput { repr_state, reward }
}

/// State to representation for Muz - single batch
pub fn representation_evaluator_single<G: Dimension, H: Dimension>(
    session: &Session,
    graph: &Graph,
    hidden_shape: H,
    state: Array<f32, G>,
) -> Array<f32, H> {
    let board_tensor = Tensor::new(
        &state
            .raw_dim()
            .insert_axis(Axis(0))
            .as_array_view()
            .to_slice()
            .unwrap()
            .iter()
            .map(|i| *i as u64)
            .collect::<Vec<u64>>(),
    )
    .with_values(&state.into_raw_vec())
    .unwrap();

    let repr_board_tensor = tf::call_representation(session, graph, &board_tensor);
    tensor_to_ndarray(repr_board_tensor, hidden_shape)
}
