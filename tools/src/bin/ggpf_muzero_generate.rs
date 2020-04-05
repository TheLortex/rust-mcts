#![allow(non_snake_case)]

use ggpf::deep::self_play::GameHistoryEntry;
use ggpf::game::meta::with_history::*;
use ggpf::game::openai::{Gym, GymBuilder};
use ggpf::game::{
    breakthrough::{Breakthrough, BreakthroughBuilder},
    *,
};
use ggpf::policies::mcts::{muz, puct::PUCTSettings};
use ggpf::settings;

use ndarray::Dim;
use std::path::Path;
use tokio::runtime;
use tokio::sync::mpsc;
use typenum::U2;


const MODEL_PATH: &str = "models/mu-breakthrough/";
type G = WithHistory<Breakthrough, U2>;


//const MODEL_PATH: &str = "models/mu-atari/";
//type G = Gym;

fn main() {
    let mut threaded_rt = runtime::Builder::new()
        .threaded_scheduler()
        .enable_all()
        .core_threads(9)
        .build()
        .unwrap();

    threaded_rt.block_on(run());
}

async fn run() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("Zerol generate: starting!");

    /* check that model exists. */
    if !Path::new(MODEL_PATH).exists() {
        println!("Couldn't find model at {}", MODEL_PATH);
        return;
    };

    // Game builder.
    //let game_builder = GymBuilder {
    //    address: "localhost:1337".into(),
    //    game_name: "Breakout-v0".into(),
    //    render: false,
    //};
    //let state: G = SingleplayerGameBuilder::create(&game_builder).await;
    let game_builder = WithHistoryGB::<BreakthroughBuilder, U2>::new(&BreakthroughBuilder {});
    let state: G = game_builder.create(G::players()[0]).await;

    let ft = state.get_features();

    let repr_dimension = Dim(settings::MUZ_BT_SHAPE);

    let config = muz::MuZeroConfig {
        puct: PUCTSettings {
            DECODE_VALUE: true,
            ..PUCTSettings::default()
        },
        action_shape: G::action_dimension(&ft),
        repr_board_shape: repr_dimension,
        board_shape: G::state_dimension(&ft),
        networks_path: MODEL_PATH.into(),
        batch_size: settings::GPU_BATCH_SIZE,
        watch_models: true,
    };

    let mut fm = ggpf::deep::file_manager::FileManager::new("./fifo");

    // Game channel.
    let (tx_games, mut rx_games) =
        mpsc::channel::<GameHistoryEntry<WithHistory<Breakthrough, U2>>>(128);

    // Game generator.
    let game_gen = tokio::spawn(ggpf::deep::self_play::muzero_game_generator(
        config,
        game_builder,
        tx_games,
    ));

    let game_writer = tokio::spawn(async move {
        while let Some(game) = rx_games.recv().await {
            fm.append(game);
        }
    });

    game_gen.await.unwrap();
    game_writer.await.unwrap();
}
