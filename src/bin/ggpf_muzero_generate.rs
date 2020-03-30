#![allow(non_snake_case)]

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::RwLock;
use std::{thread};
use std::sync::mpsc;
use ndarray::Dim;


const MODEL_PATH: &str = "models/mu-breakthrough/";

use ggpf::game::{
    breakthrough::{Breakthrough, BreakthroughBuilder},
};
use ggpf::game::meta::with_history::*;
use ggpf::policies::mcts::puct::PUCTSettings;
use ggpf::deep::self_play::GameHistoryEntry;
use ggpf::deep::tf;
use ggpf::deep::filemanager;
use ggpf::settings;

use typenum::U2;

fn main() {
    run()
}

fn run() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("Zerol generate: starting!");

    /* check that model exists. */
    if !Path::new(MODEL_PATH).exists() {
        println!("Couldn't find model at {}", MODEL_PATH);
        return;
    };

    let prediction_path = format!("{}{}", MODEL_PATH, "pv");
    let dynamics_path = format!("{}{}", MODEL_PATH, "dyn");
    let representation_path = format!("{}{}", MODEL_PATH, "state");
    
    let prediction_tensorflow       = Arc::new((AtomicBool::new(false), RwLock::new(tf::load_model(&prediction_path))));
    let dynamics_tensorflow         = Arc::new((AtomicBool::new(false), RwLock::new(tf::load_model(&dynamics_path))));
    let representation_tensorflow   = Arc::new((AtomicBool::new(false), RwLock::new(tf::load_model(&representation_path))));


    let mut fm = ggpf::deep::filemanager::FileManager::new(
        "./fifo",
    );

    /*
     * Watches for change in the model, and reload when needed.
     */

    filemanager::watch_model(prediction_tensorflow.clone(), prediction_path);

    filemanager::watch_model(dynamics_tensorflow.clone(), dynamics_path);

    filemanager::watch_model(representation_tensorflow.clone(), representation_path);

    // Game channel.
    let (tx_games, rx_games) = mpsc::sync_channel::<GameHistoryEntry<WithHistory<Breakthrough,U2>>>(1024);

    // Game builder.
    let game_builder = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});

    // Game generator.
    let game_gen = thread::spawn(move || ggpf::deep::self_play::muzero_game_generator(
        (PUCTSettings::default(), Dim(settings::MUZ_BT_SHAPE)),
        game_builder,
        prediction_tensorflow,
        dynamics_tensorflow,
        representation_tensorflow,
        tx_games,
    ));

    let game_writer = thread::spawn(move || {
        while let Some(game) = rx_games.recv().ok() {
            fm.append(game);
        }
    });

    game_gen.join().unwrap();
    game_writer.join().unwrap();
}
