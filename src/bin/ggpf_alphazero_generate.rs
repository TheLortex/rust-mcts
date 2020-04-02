#![allow(non_snake_case)]

use ggpf::deep::file_manager;
use ggpf::deep::self_play::GameHistoryEntry;
use ggpf::game::breakthrough::{Breakthrough, BreakthroughBuilder};
use ggpf::game::meta::with_history::*;
use ggpf::game::*;
use ggpf::policies::mcts::puct::{AlphaZeroConfig, PUCTSettings};

use tokio::runtime;
use tokio::sync::mpsc;
use typenum::U2;

const MODEL_PATH: &str = "models/alpha-breakthrough/";

fn main() {
    let mut threaded_rt = runtime::Builder::new()
        .threaded_scheduler()
        .enable_all()
        .core_threads(8)
        .build()
        .unwrap();

    threaded_rt.block_on(run());
}

type G = Breakthrough;

async fn run() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("AlphaZero generate: starting!");

    // Game builder.
    let game_builder = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});

    let state: WithHistory<G, _> = game_builder.create(G::players()[0]);

    let board_shape = state.state_dimension();
    let action_shape = G::action_dimension();

    log::debug!("Action: {:?}", action_shape);
    log::debug!("Board: {:?}", board_shape);

    let config = AlphaZeroConfig {
        action_shape,
        board_shape,
        puct: PUCTSettings {
            DECODE_VALUE: false,
            ..PUCTSettings::default()
        },
        network_path: MODEL_PATH.into(),
        watch_models: true,
        batch_size: ggpf::settings::GPU_BATCH_SIZE,
    };

    // Game channel.
    let (tx_games, mut rx_games) =
        mpsc::channel::<GameHistoryEntry<WithHistory<Breakthrough, U2>>>(1024);

    // Game generator.
    let game_gen = tokio::spawn(ggpf::deep::self_play::alphazero_game_generator(
        config,
        game_builder,
        tx_games,
    ));

    // Game writer.
    let mut fm = file_manager::FileManager::new("./fifo");
    let game_writer = tokio::spawn(async move {
        while let Some(game) = rx_games.recv().await {
            fm.append(game);
        }
    });

    game_writer.await.unwrap();
    game_gen.await.unwrap();
}
