#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::BufReader;
use std::iter::FromIterator;
use std::path::Path;
use std::process::exit;

use tensorflow::{Code, Graph, Session, SessionOptions};

const MODEL_PATH: &str = "models/breakthrough";

use zerol::game::breakthrough::{Breakthrough, K, BreakthroughBuilder, Color, Move};
use zerol::game::{BaseGame, MultiplayerGame, MultiplayerGameBuilder};
use zerol::game;
use zerol::policies::puct::PUCT;
use zerol::policies::flat::Random;
use zerol::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};

use nix::unistd::mkfifo;
use std::io::SeekFrom;
use std::marker::PhantomData;


use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::Array;

fn self_play_match<F: Fn(Color, &[&Breakthrough]) -> (Array<f32, <Breakthrough as game::Feature>::ActionDim>, f32)>(
    evaluate: &F,
    gb: &BreakthroughBuilder
) -> (Color, Vec<(Breakthrough, HashMap<Move, f32>)>) {
    let mut game: Breakthrough = gb.create(Color::random());
    
    let puct = PUCT {
        C_PUCT: 4.,
        evaluate,
        _g: PhantomData,
    };
    let mut p1 = puct.create(Color::Black);
    let mut p2 = puct.create(Color::White);
    //let mut p1 = MultiplayerPolicyBuilder::<Breakthrough>::create(&Random {}, Color::Black);
    //let mut p2 = MultiplayerPolicyBuilder::<Breakthrough>::create(&Random {}, Color::White);

    let mut history = vec![];

    while { game.winner().is_none() } {
        let policy = if game.turn() == Color::Black { // : &mut dyn MultiplayerPolicy<Breakthrough>
            &mut p1
        } else {
            &mut p2
        };

        let action = policy.play(&game);
        let game_node = policy.0.tree.get(&game.hash()).unwrap();
        let monte_carlo_distribution = HashMap::from_iter(
            game_node
                .moves
                .iter()
                .map(|(k, v)| (*k, v.N_a / game_node.count)),
        );
        history.push((game.clone(), monte_carlo_distribution));

        game.play(&action);
    }

    (game.winner().unwrap(), history)
}

struct FileManager {
    f: File,
}

use nix::sys::stat;
use std::mem::transmute;

use zerol::game::Feature;

impl FileManager {
    fn new(path: &str) -> Self {
        match mkfifo(path, stat::Mode::S_IRWXU) {
            Ok(_) => println!("Created FIFO {}.", path),
            Err(_) => println!("FIFO already exists."),
        };

        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .expect("Unable to open file.");
        FileManager { f }
    }

    fn append(&mut self, board: &Breakthrough, policy: &HashMap<Move, f32>, value: &f32) {
        let vec_board = board.state_to_feature(board.turn()).into_raw_vec();
        
        let vec_policy = Breakthrough::moves_to_feature(policy).into_raw_vec();

        let toser = (vec_board, vec_policy, value);
        let ser = serde_pickle::to_vec(&toser, true).unwrap();
        self.f.write(&ser.len().to_be_bytes()).expect(":c");
        self.f.write_all(&ser).expect("Could not write file..");
    }
}

fn main() {
    let mut threaded_rt = tokio::runtime::Builder::new()
        .threaded_scheduler()
        .core_threads(8)
        .build().unwrap();

    threaded_rt.block_on(run())
}

use notify::event::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::time::Duration;
use std::sync::Mutex;
use std::sync::Arc;

use zerol::misc::breakthrough_evaluator;
use rayon::prelude::*;
use rayon::iter::repeat;

use std::sync::RwLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use std::{thread, time};

use tensorflow;
use tokio;

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
    let fm_mtx = Arc::new(Mutex::new(FileManager::new("./fifo")));

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
            for (board, policy) in history.iter() {
                let value = if board.turn() == winner { 1.0 } else { 0.0 };
                fm.append(board, policy, &value);
            }
        }
    });

    game_gen.await.unwrap();
    game_writer.await.unwrap();
}
