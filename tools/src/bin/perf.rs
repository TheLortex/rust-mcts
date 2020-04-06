//! # PERF - benchmark game generation performance.
//!
//! Usage: `cargo run --release --bin perf`
//!
//! Performance is tested on 5x5 breakthrough/PUCT.

#![allow(non_snake_case)]

use ggpf::deep::evaluator::PredictionEvaluatorChannel;
use ggpf::deep::tf;
use ggpf::game::breakthrough::{Breakthrough, BreakthroughBuilder};
use ggpf::game::meta::with_history::*;
use ggpf::game::*;

use ndarray::Dimension;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::runtime;
use tokio::sync::mpsc;

/// Model location.
const MODEL_PATH: &str = "data/alpha-breakthrough-5/model/";
/// Game type.
type G = WithHistory<Breakthrough>;
/// Entry point.
fn main() {
    let mut threaded_rt = runtime::Builder::new()
        .threaded_scheduler()
        .enable_all()
        .core_threads(8)
        .build()
        .unwrap();

    threaded_rt.block_on(run());
}

use indicatif::{ProgressBar, ProgressStyle};

/// Batch size per evaluator.
const GPU_BATCH_SIZE: usize = 128;
/// Number of game generators per evaluator.
const N_GENERATORS: usize = 256;
/// Number of evaluators.
const N_EVALUATORS: usize = 4;

/// Run performance test with hardcoded configuration.
async fn run() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("AlphaZero generate: starting!");

    /* check that model exists. */
    if !Path::new(MODEL_PATH).exists() {
        println!("Couldn't find model at {}", MODEL_PATH);
        return;
    };

    // Load neural network
    let prediction_tensorflow = Arc::new((
        AtomicBool::new(false),
        RwLock::new(tf::load_model(&MODEL_PATH)),
    ));

    // Game builder.
    let game_builder = WithHistoryGB::new(BreakthroughBuilder { size: 5 }, 2);

    let breakthrough: G = game_builder.create(Breakthrough::players()[0]).await;

    let ft = breakthrough.get_features();
    let board_size = G::state_dimension(&ft).size();
    let action_size = G::action_dimension(&ft).size();

    let indicator_bar = ProgressBar::new_spinner();
    indicator_bar.set_style(
        ProgressStyle::default_spinner()
            .template("[{spinner}] {wide_bar} {pos} steps generated ({elapsed_precise})"),
    );
    indicator_bar.enable_steady_tick(200);

    let bar_box = Arc::new(Box::new(indicator_bar));

    let mut jh = vec![];

    for _ in 0..N_EVALUATORS {
        let (pred_tx, pred_rx) = mpsc::channel::<PredictionEvaluatorChannel>(2 * GPU_BATCH_SIZE);

        for _ in 0..N_GENERATORS {
            let ptx = pred_tx.clone();
            let bt = breakthrough.clone();
            tokio::spawn(async move {
                loop {
                    ggpf::deep::evaluator::prediction(
                        ptx.clone(),
                        Breakthrough::players()[0],
                        &bt,
                        1,
                    )
                    .await;
                }
            });
        }

        let prediction_tensorflow = prediction_tensorflow.clone();

        let bb = bar_box.clone();

        jh.push(tokio::spawn(ggpf::deep::evaluator::prediction_task(
            GPU_BATCH_SIZE,
            board_size,
            action_size,
            1,
            prediction_tensorflow,
            pred_rx,
            Some(bb),
        )));
    }

    for i in jh {
        i.await.unwrap()
    }
}
