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


const MODEL_PATH: &str = "models/breakthrough";

use zerol::game::{
    breakthrough::{Breakthrough, BreakthroughBuilder},
    Game,
};
use zerol::game::meta::with_history::*;
use zerol::policies::mcts::puct::PUCTSettings;
use zerol::r#async::GameHistoryChannel;

use typenum::U2;

fn main() {
    run()
}

fn run() {
    /* check that model exists. */
    if !Path::new(MODEL_PATH).exists() {
        println!("Couldn't find model at {}", MODEL_PATH);
        return;
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
    let session = Session::from_saved_model(&options, &["serve"], &mut graph, MODEL_PATH).unwrap();

    let is_writing = Arc::new(AtomicBool::new(false));
    let graph_and_session = Arc::new(RwLock::new((graph, session)));
    let g_and_s = graph_and_session.clone();
    let i_w = is_writing.clone();

    //graph.operation_iter().map(|o| println!("{:?}", o.name().unwrap())).collect::<()>();
    let fm_mtx = Arc::new(Mutex::new(zerol::misc::filemanager::FileManager::new(
        "./fifo",
    )));

    /*
     * Watches for change in the model, and reload when needed.
     */
    let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |x: Result<Event, _>| {
        if let Ok(x) = x {
            if x.kind == EventKind::Access(AccessKind::Close(AccessMode::Write)) {
                println!("Updating model..\n");
                i_w.store(true, Ordering::Relaxed);
                let mut locked_mutex = g_and_s.write().unwrap();
                i_w.store(false, Ordering::Relaxed);
                let (ref _graph, ref mut session_m) = *locked_mutex;
                session_m.close().expect("Unable to close the session.");

                let mut graph = Graph::new();
                let mut options = SessionOptions::new();
                options.set_config(&configuration_buf).unwrap();

                let session =
                    Session::from_saved_model(&options, &["serve"], &mut graph, MODEL_PATH)
                        .unwrap();

                *locked_mutex = (graph, session);
                println!("Sucessfully updated model.");
            }
        }
    })
    .unwrap();
    watcher
        .watch(MODEL_PATH, RecursiveMode::NonRecursive)
        .unwrap();

    let (tx_games, rx_games) = mpsc::sync_channel::<GameHistoryChannel<WithHistory<Breakthrough,U2>>>(1024);

    let game_builder = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});

    let game_gen = thread::spawn(move || zerol::r#async::game_generator(
        PUCTSettings::default(),
        game_builder,
        graph_and_session,
        tx_games,
    ));

    let game_writer = thread::spawn(move || {
        while let Some((winner, history)) = rx_games.recv().ok() {
            while is_writing.load(Ordering::Relaxed) {
                thread::sleep(time::Duration::from_millis(1));
            }

            let fm_mtx = fm_mtx.clone();
            let mut fm = fm_mtx.lock().unwrap();
            
            for (board, policy) in history.iter() {
                let value = if board.turn() == winner { 1.0 } else { 0.0 };
                fm.append(&board, policy, value);
            }
        }
    });

    game_gen.join().unwrap();
    game_writer.join().unwrap();
}
