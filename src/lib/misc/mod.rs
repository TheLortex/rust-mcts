use crate::game;

use tensorflow::{Graph, Session, SessionRunArgs, Tensor};
use ndarray::{Dimension, Array, ArrayBase, Axis};

pub mod filemanager;

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


    let (policy_tensor, value_tensor) = tensorflow_call(session, graph, &board_tensor);
    let policy = ArrayBase::from_shape_vec(G::action_dimension(), (&policy_tensor).to_vec()).unwrap();
    let value = value_tensor[0];
    (policy, value)
}
