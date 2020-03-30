#![allow(non_snake_case)]

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;

use ggpf::deep::filemanager;
use ggpf::deep::self_play::GameHistoryEntry;
use ggpf::deep::tf;
use ggpf::game::breakthrough::{Breakthrough, BreakthroughBuilder};
use ggpf::game::meta::with_history::*;
use ggpf::policies::mcts::puct::PUCTSettings;

use typenum::U2;

fn main() {
    run()
}

const MODEL_PATH: &str = "models/alpha-breakthrough/";

fn run() {
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

    /*
     * Watches for change in the model, and reload when needed.
     */
    let p_tf = prediction_tensorflow.clone();
    filemanager::watch_model(p_tf, String::from("./models/alpha-breakthrough"));

    // Game channel.
    let (tx_games, rx_games) =
        mpsc::sync_channel::<GameHistoryEntry<WithHistory<Breakthrough, U2>>>(1024);

    // Game builder.
    let game_builder = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});

    // Game generator.
    let game_gen = thread::spawn(move || {
        ggpf::deep::self_play::alphazero_game_generator(
            PUCTSettings::default(),
            game_builder,
            prediction_tensorflow,
            tx_games,
        )
    });

    // Game writer.
    let mut fm = filemanager::FileManager::new("./fifo");
    let game_writer = thread::spawn(move || {
        while let Some(game) = rx_games.recv().ok() {
            fm.append(game);
        }
    });

    game_gen.join().unwrap();
    game_writer.join().unwrap();
}
