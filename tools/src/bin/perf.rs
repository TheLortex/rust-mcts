#![allow(non_snake_case)]

use ggpf::deep::evaluator::PredictionEvaluatorChannel;
use ggpf::deep::tf;
use ggpf::game::breakthrough::{Breakthrough, BreakthroughBuilder};
use ggpf::game::meta::with_history::*;
use ggpf::game::*;
use ggpf::settings;

use ndarray::Dimension;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::runtime;
use tokio::sync::mpsc;
use typenum::U2;

const MODEL_PATH: &str = "models/alpha-breakthrough/";

type G = WithHistory<Breakthrough, U2>;
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
    let game_builder = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});

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

    for _ in 0..4 {
        let (pred_tx, pred_rx) =
            mpsc::channel::<PredictionEvaluatorChannel>(2 * settings::GPU_BATCH_SIZE);

        for _ in 0..256 {
            let ptx = pred_tx.clone();
            let bt = breakthrough.clone();
            tokio::spawn(async move {
                loop {
                    ggpf::deep::evaluator::prediction(
                        ptx.clone(),
                        Breakthrough::players()[0],
                        &bt,
                        false,
                    )
                    .await;
                }
            });
        }

        let prediction_tensorflow = prediction_tensorflow.clone();

        let bb = bar_box.clone();

        jh.push(tokio::spawn(ggpf::deep::evaluator::prediction_task(
            settings::GPU_BATCH_SIZE,
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

/*

BATCH SIZE   -   N GENERATORS   - N EVALUATORS -   STEPS/10S
    1                  1               1              9300 *
    1                  1               2             18000 *
    1                  1               4             27000 *
    1                  1               6             27500 *
    1                  4               1             15000 *
    1                 16               1              40K  *
    4                  4               1             19000 *
    4                 16               1             30000 *
   16                 16               1              2833 *
   16                 64               1             100K  *
   16                 64               4             216K  *
   64                 16               1              10K  *
   64                256               4             365K  *
  256                256               4             422K  *
  256               1024               4             2M
  512               1024               4             415K  *
  256               2048               2             380K  *
  128                256               8             376K  *
*/
