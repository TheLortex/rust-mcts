use crate::game;
use crate::settings::GPU_BATCH_SIZE;

use tensorflow::{Graph, Session, SessionRunArgs, Tensor};
use ndarray::{Dimension, Array, ArrayBase, Axis};

const SUPPORT_SIZE: isize = 1;
const SUPPORT_SHAPE: isize = 2*SUPPORT_SIZE+1;

fn sign(x: f32) -> f32 {
    if x > 0. {
        1.
    } else if x == 0. {
        0.
    } else {
        -1.
    }
}

fn support_to_value(support: &Tensor<f32>) -> Tensor<f32> {
    let mut res = Tensor::new(&[GPU_BATCH_SIZE as u64]);

    for i in 0..GPU_BATCH_SIZE {
        let value: f32 = (-SUPPORT_SIZE..SUPPORT_SIZE+1).enumerate().map(|(j, v)| support[(SUPPORT_SHAPE as usize)*i+j]*(v as f32)).sum();
        let value: f32 = sign(value) * ((((1. + 4. * 0.001 * (value.abs() + 1. + 0.001)).sqrt() - 1.) / (2. * 0.001)).powi(2) - 1.); 
    
        res[i] = value;
    }
    res
}

/// Use prediction network inference.
pub fn call_prediction(
    session: &Session,
    graph: &Graph,
    board: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    
    let board_op = graph
        .operation_by_name_required("serving_default_board")
        .unwrap();
    let output_op = graph
        .operation_by_name_required("StatefulPartitionedCall")
        .unwrap();
    let mut args = SessionRunArgs::new();
    args.add_feed(&board_op, 0, board);

    let policy_req = args.request_fetch(&output_op, 0);
    let value_req = args.request_fetch(&output_op, 1);
    session.run(&mut args).unwrap();

    let policy_tensor: Tensor<f32> = args.fetch(policy_req).unwrap();
    let value_tensor: Tensor<f32> = args.fetch(value_req).unwrap();
    (policy_tensor, support_to_value(&value_tensor))
}

/// Use dynamics network inference.
pub fn call_dynamics(
    session: &Session,
    graph: &Graph,
    board: &Tensor<f32>,
    action: &Tensor<f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    
    let board_op = graph
        .operation_by_name_required("serving_default_board")
        .unwrap();
    let action_op = graph
        .operation_by_name_required("serving_default_action")
        .unwrap();
    let output_op = graph
        .operation_by_name_required("StatefulPartitionedCall")
        .unwrap();
    let mut args = SessionRunArgs::new();
    args.add_feed(&board_op, 0, board);
    args.add_feed(&action_op, 0, action);

    let reward_req = args.request_fetch(&output_op, 1);
    let next_board_req = args.request_fetch(&output_op, 0);
    session.run(&mut args).unwrap();

    let reward_tensor: Tensor<f32> = args.fetch(reward_req).unwrap();
    let next_board_tensor: Tensor<f32> = args.fetch(next_board_req).unwrap();
    (support_to_value(&reward_tensor), next_board_tensor)
}

/// Use representation network inference.
pub fn call_representation(
    session: &Session,
    graph: &Graph,
    board: &Tensor<f32>,
) -> Tensor<f32> {
    
    let board_op = graph
        .operation_by_name_required("serving_default_board")
        .unwrap();
    let output_op = graph
        .operation_by_name_required("StatefulPartitionedCall")
        .unwrap();
    let mut args = SessionRunArgs::new();
    args.add_feed(&board_op, 0, board);

    let repr_board_req = args.request_fetch(&output_op, 0);
    session.run(&mut args).unwrap();

    let repr_board_tensor: Tensor<f32> = args.fetch(repr_board_req).unwrap();
    repr_board_tensor
}

/// Evaluates a game state for PUCT.
pub fn game_evaluator<G: game::Feature>(
    session: &Session,
    graph: &Graph,
    pov: G::Player,
    board: &G,
) -> (Array<f32, G::ActionDim>, f32) {
    let input_dimensions = board.state_dimension();

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


    let (policy_tensor, value_tensor) = call_prediction(session, graph, &board_tensor);
    let policy = ArrayBase::from_shape_vec(G::action_dimension(), (&policy_tensor).to_vec()).unwrap();
    let value = value_tensor[0];
    (policy, value)
}
