use super::game::breakthrough::{Breakthrough, Move};
use super::game::BaseGame;

use std::collections::HashMap;
use std::iter::FromIterator;

use tensorflow::{Graph, Session, SessionRunArgs, Tensor};

const K: usize  = 8; //TODO
pub fn evaluator(session: &Session, graph: &Graph, board: &Breakthrough) -> (HashMap<Move, f32>, f32) {
    let K_ = K as u64;
    let mut board_tensor: Tensor<f32> = Tensor::new(&[1, 2 * K_ * K_ + 1]);
    for (j, item) in board.serialize().iter().enumerate() {
        board_tensor[j] = *item;
    }

    let board_op = graph
        .operation_by_name_required("serving_default_board")
        .unwrap();
    let output_op = graph
        .operation_by_name_required("StatefulPartitionedCall")
        .unwrap();
    let mut args = SessionRunArgs::new();
    args.add_feed(&board_op, 0, &board_tensor);

    let policy_req = args.request_fetch(&output_op, 0);
    let value_req = args.request_fetch(&output_op, 1);
    session.run(&mut args).unwrap();

    let policy_tensor: Tensor<f32> = args.fetch(policy_req).unwrap();
    let value_tensor: Tensor<f32> = args.fetch(value_req).unwrap();

    let policy = HashMap::from_iter(
        board
            .possible_moves()
            .into_iter()
            .map(|m| (*m, policy_tensor[m.serialize()])),
    );
    let value = value_tensor[0];
    (policy, value)
}