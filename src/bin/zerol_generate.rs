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

use tensorflow::{Code, Graph, Session, SessionOptions, Status};

const MODEL_PATH: &str = "models/breakthrough";

use zerol::game::breakthrough::{Breakthrough, K, BreakthroughBuilder, Color, Move};
use zerol::game::{BaseGame, MultiplayerGame, MultiplayerGameBuilder};
use zerol::policies::puct::PUCT;
use zerol::policies::flat::Random;
use zerol::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};

use nix::unistd::mkfifo;
use std::io::SeekFrom;
use std::marker::PhantomData;


use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

fn self_play_match<F: Fn(&Breakthrough) -> (HashMap<Move, f32>, f32)>(
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
        let vec_board = board.serialize();
        let mut vec_policy = vec![0.; 3 * K * K];
        for (k, v) in policy.iter() {
            vec_policy[k.serialize()] = *v;
        }

        let toser = (vec_board, vec_policy, value);
        let ser = serde_pickle::to_vec(&toser, true).unwrap();
        self.f.write(&ser.len().to_be_bytes()).expect(":c");
        self.f.write_all(&ser).expect("Could not write file..");
    }
}

fn main() {
    exit(match run() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
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

fn run() -> Result<(), Box<dyn Error>> {
    /* check that model exists. */
    if !Path::new(MODEL_PATH).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python zerol.py' to generate \
                     {} and try again.",
                    MODEL_PATH
                ),
            )
            .unwrap(),
        ));
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
    )?;

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

    let gb = BreakthroughBuilder {};

    let bar = ProgressBar::new_spinner();
    bar.set_style(ProgressStyle::default_spinner().template("[{spinner}] {wide_bar} {pos} games generated ({elapsed_precise})"));
    bar.enable_steady_tick(200);

    repeat(()).map(|_| {
        let (winner,history) = {
            let (ref graph, ref session) = *graph_and_session.read().unwrap();
            self_play_match(&(|board| breakthrough_evaluator(session, graph, board)), &gb)
        };
        {
            while is_writing.load(Ordering::Relaxed) {
                thread::sleep(time::Duration::from_millis(1));
            };
            let fm_mtx = fm_mtx.clone();
            let mut fm = fm_mtx.lock().unwrap();
            for (board, policy) in history.iter() {
                let value = if board.turn() == winner { 1.0 } else { 0.0 };
                //fm.append(board, policy, &value);
            }
        }
        bar.inc(1);
    }).collect::<()>();
    Ok(())
}
