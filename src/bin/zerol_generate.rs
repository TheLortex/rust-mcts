#![allow(non_snake_case)]

use tensorflow::{Graph, Session, SessionOptions};
use notify::event::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::path::Path;
use std::{thread, time};
use tensorflow;
use tokio;

const MODEL_PATH: &str = "models/breakthrough";

use zerol::game::{MultiplayerGame, breakthrough::Breakthrough};
use zerol::settings;

fn main() {
    let mut threaded_rt = tokio::runtime::Builder::new()
        .threaded_scheduler()
        .core_threads(8)
        .build().unwrap();

    threaded_rt.block_on(run())
}


async fn run() {
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
    let session = Session::from_saved_model(
        &options,
        &["serve"],
        &mut graph,
        MODEL_PATH,
    ).unwrap();

    let is_writing = Arc::new(AtomicBool::new(false));
    let graph_and_session = Arc::new(RwLock::new((graph,session)));
    let g_and_s = graph_and_session.clone();
    let i_w = is_writing.clone();

    //graph.operation_iter().map(|o| println!("{:?}", o.name().unwrap())).collect::<()>();
    let fm_mtx = Arc::new(Mutex::new(zerol::misc::filemanager::FileManager::new("./fifo")));

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

                let session = Session::from_saved_model(
                    &options,
                    &["serve"],
                    &mut graph,
                    MODEL_PATH,
                ).unwrap();

                *locked_mutex = (graph, session);
                println!("Sucessfully updated model.");
            }
        }
    })
    .unwrap();
    watcher
        .watch(MODEL_PATH, RecursiveMode::NonRecursive)
        .unwrap();


    let (tx_games, mut rx_games) = tokio::sync::mpsc::channel(1024);
    
    let game_gen = tokio::spawn(zerol::r#async::game_generator(graph_and_session, tx_games));

    let game_writer = tokio::spawn(async move {
        while let Some((winner, history)) = rx_games.recv().await {
            while is_writing.load(Ordering::Relaxed) {
                thread::sleep(time::Duration::from_millis(1));
            };



            let fm_mtx = fm_mtx.clone();
            let mut fm = fm_mtx.lock().unwrap();
            let mut board_history: Vec<Breakthrough> = vec![];
            for (board, policy) in history.iter() {
                while board_history.len() < settings::DEFAULT_N_HISTORY_PUCT-1 {
                    board_history.push(board.clone());
                };
                board_history.push(board.clone());
                let value = if board.turn() == winner { 1.0 } else { 0.0 };
                fm.append(&board_history, policy, &value);
            }
        }
    });

    game_gen.await.unwrap();
    game_writer.await.unwrap();
}
