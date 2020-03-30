#![allow(non_snake_case)]

use notify::event::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::{thread, time};
use std::sync::mpsc;

use tensorflow::{Graph, Session, SessionOptions};


const MODEL_PATH: &str = "models/breakthrough/";

use ggpf::game::{
    breakthrough::{Breakthrough, BreakthroughBuilder},
};
use ggpf::game::meta::with_history::*;
use ggpf::policies::mcts::puct::PUCTSettings;
use ggpf::deep::self_play::GameHistoryEntry;

use typenum::U2;

fn main() {
    run()
}

fn tf_load_model(path: &str) -> (Graph, Session) {

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

fn run() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("Zerol generate: starting!");

    /* check that model exists. */
    if !Path::new(MODEL_PATH).exists() {
        println!("Couldn't find model at {}", MODEL_PATH);
        return;
    };
    
    let prediction_tensorflow       = Arc::new(RwLock::new(tf_load_model(&format!("{}{}", MODEL_PATH, "pv"))));
    let dynamics_tensorflow         = Arc::new(RwLock::new(tf_load_model(&format!("{}{}", MODEL_PATH, "dyn"))));
    let representation_tensorflow   = Arc::new(RwLock::new(tf_load_model(&format!("{}{}", MODEL_PATH, "state"))));

    let is_writing = Arc::new(AtomicBool::new(false));
    let i_w = is_writing.clone();

    //graph.operation_iter().map(|o| println!("{:?}", o.name().unwrap())).collect::<()>();
    let fm_mtx = Arc::new(Mutex::new(ggpf::deep::filemanager::FileManager::new(
        "./fifo",
    )));

    /*
     * Watches for change in the model, and reload when needed.
     */
    let prediction_tf_for_update = prediction_tensorflow.clone();
    let dynamics_tf_for_update = dynamics_tensorflow.clone();
    let representation_tf_for_update = representation_tensorflow.clone();

    let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |x: Result<Event, _>| {
        if let Ok(x) = x {
            if x.kind == EventKind::Access(AccessKind::Close(AccessMode::Write)) {
                println!("Updating model..\n");
                i_w.store(true, Ordering::Relaxed);
                let mut prediction_tensorflow = prediction_tf_for_update.write().unwrap();
                let mut dynamics_tensorflow = dynamics_tf_for_update.write().unwrap();
                let mut representation_tensorflow = representation_tf_for_update.write().unwrap();
                i_w.store(false, Ordering::Relaxed);
                let (ref _graph, ref mut session_m) = *prediction_tensorflow;
                session_m.close().expect("Unable to close the prediction session.");
                let (ref _graph, ref mut session_m) = *dynamics_tensorflow;
                session_m.close().expect("Unable to close the dynamics session.");
                let (ref _graph, ref mut session_m) = *representation_tensorflow;
                session_m.close().expect("Unable to close the representation session.");

                *prediction_tensorflow = tf_load_model(&format!("{}{}", MODEL_PATH, "pv"));
                *dynamics_tensorflow = tf_load_model(&format!("{}{}", MODEL_PATH, "dyn"));
                *representation_tensorflow = tf_load_model(&format!("{}{}", MODEL_PATH, "state"));
                println!("Sucessfully updated model.");
            }
        }
    })
    .unwrap();
    watcher
        .watch(MODEL_PATH, RecursiveMode::NonRecursive)
        .unwrap();

    let (tx_games, rx_games) = mpsc::sync_channel::<GameHistoryEntry<WithHistory<Breakthrough,U2>>>(1024);

    let game_builder = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});

    let game_gen = thread::spawn(move || ggpf::deep::self_play::game_generator(
        PUCTSettings::default(),
        game_builder,
        prediction_tensorflow,
        dynamics_tensorflow,
        representation_tensorflow,
        tx_games,
    ));

    let game_writer = thread::spawn(move || {
        let fm_mtx = fm_mtx.clone();
        while let Some(game) = rx_games.recv().ok() {
            while is_writing.load(Ordering::Relaxed) {
                thread::sleep(time::Duration::from_millis(1));
            }

            let mut fm = fm_mtx.lock().unwrap();
            fm.append(game);
        }
    });

    game_gen.join().unwrap();
    game_writer.join().unwrap();
}
