use super::game;
use super::game::breakthrough::{Breakthrough, Move, K};
use super::game::BaseGame;

use std::collections::HashMap;
use std::iter::FromIterator;

use tensorflow::{Graph, Session, SessionRunArgs, Tensor};

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

use ndarray::{Dimension, Array, ArrayBase};

pub fn breakthrough_evaluator<G: game::Feature>(
    session: &Session,
    graph: &Graph,
    pov: G::Player,
    board_history: &[&G],
) -> (Array<f32, G::ActionDim>, f32) {
    let input_dimensions = G::state_dimension();
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

    let (policy_tensor, value_tensor) = tensorflow_call(session, graph, &board_tensor);
    let policy = ArrayBase::from_shape_vec(G::action_dimension(), (&policy_tensor).to_vec()).unwrap();
    let value = value_tensor[0];
    (policy, value)
}
