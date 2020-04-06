use std::path::Path;
use std::sync::{atomic::AtomicBool, Arc, RwLock};
use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor};

/// Access to a TF model behind Arc and RwLock
/// the AtomicBool is here to indicate the file loader's intention
/// to access the lock.

pub type ThreadSafeModel = Arc<(AtomicBool, RwLock<(Graph, Session)>)>;

fn sign(x: f32) -> f32 {
    if x > 0. {
        1.
    } else if x == 0. {
        0.
    } else {
        -1.
    }
}

/// Converts a suport encoding of scalar to the corresponding value.
pub fn support_to_value(
    support: &Tensor<f32>,
    batch_size: usize,
    support_size: usize,
) -> Tensor<f32> {
    let mut res = Tensor::new(&[batch_size as u64]);

    for i in 0..batch_size {
        let value: f32 = (-(support_size as isize)..(support_size as isize + 1))
            .enumerate()
            .map(|(j, v)| support[(2 * support_size + 1) * i + j] * (v as f32))
            .sum();
        let value: f32 = sign(value)
            * ((((1. + 4. * 0.001 * (value.abs() + 1. + 0.001)).sqrt() - 1.) / (2. * 0.001))
                .powi(2)
                - 1.);

        res[i] = value;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_support_to_value() {
        let mut support = Tensor::new(&[1, 3]);

        support[0] = 1.0;
        support[1] = 0.;
        support[2] = 0.;
        println!("=> {:?}", support_to_value(&support, 1, 1).to_vec());

        support[0] = 0.;
        support[1] = 1.;
        support[2] = 0.;
        println!("=> {:?}", support_to_value(&support, 1, 1).to_vec());

        support[0] = 0.;
        support[1] = 0.;
        support[2] = 1.;
        println!("=> {:?}", support_to_value(&support, 1, 1).to_vec());
    }
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
    (policy_tensor, value_tensor)
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
    (reward_tensor, next_board_tensor)
}

/// Use representation network inference.
pub fn call_representation(session: &Session, graph: &Graph, board: &Tensor<f32>) -> Tensor<f32> {
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

/// Load a tensorflow model into a session.
pub fn load_model(path: &str) -> (Graph, Session) {
    /* check that model exists. */
    if !Path::new(path).exists() {
        log::error!("Couldn't find model at {}", path);
        panic!("");
    };

    let mut graph = Graph::new();
    let mut options = SessionOptions::new();
    /* To get configuration, use python:
     *      config = tf.ConfigProto()
     *      config.gpu_options.allow_growth = True
     *      config.SerializeToString()
     */
    let configuration_buf = [50, 2, 32, 1];
    options.set_config(&configuration_buf).unwrap();
    let session = Session::from_saved_model(&options, &["serve"], &mut graph, path).unwrap();
    (graph, session)
}
