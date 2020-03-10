use std::collections::HashMap;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::BufReader;
use std::iter::FromIterator;
use std::path::Path;
use std::process::exit;

use tensorflow::{Code, Graph, Session, SessionOptions, Status};

const MODEL_PATH: &str = "models/sample";

use zerol::game::breakthrough::{Breakthrough, Color, Move};
use zerol::game::Game;
use zerol::policies::puct::PUCTPolicy;
use zerol::policies::Policy;

use nix::unistd::mkfifo;
use std::io::SeekFrom;

fn self_play_match<F: Fn(&Breakthrough) -> (HashMap<Move, f32>, f32)>(
    evaluate: &F,
) -> (Color, Vec<(Breakthrough, HashMap<Move, f32>)>) {
    let mut game: Breakthrough = Breakthrough::new(Color::Black, ());
    let mut p1 = PUCTPolicy {
        C_PUCT: 4.,
        color: Color::Black,
        tree: HashMap::new(),
        evaluate,
    };
    let mut p2 = PUCTPolicy {
        C_PUCT: 4.,
        color: Color::White,
        tree: HashMap::new(),
        evaluate,
    };

    let mut history = vec![];

    while { game.winner().is_none() } {
        let policy = if game.turn() == Color::Black {
            &mut p1
        } else {
            &mut p2
        };

        let action = policy.play(&game);
        let game_node = policy.tree.get(&game.hash()).unwrap();
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

const K: usize = 8; // TODO

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

use zerol::misc::evaluator;

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
    let session = Session::from_saved_model(
        &SessionOptions::new(),
        &["serve"],
        &mut graph,
        MODEL_PATH,
    )?;

    let graph_and_session = Arc::new(Mutex::new((graph,session)));
    let g_and_s = graph_and_session.clone();


    //graph.operation_iter().map(|o| println!("{:?}", o.name().unwrap())).collect::<()>();
    let mut fm = FileManager::new("./fifo");

    let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |x: Result<Event, _>| {
        if let Ok(x) = x {
            if x.kind == EventKind::Access(AccessKind::Close(AccessMode::Write)) {
                
                let mut locked_mutex = g_and_s.lock().unwrap();
                let (ref _graph, ref mut session_m) = *locked_mutex;
                session_m.close().expect("Unable to close the session.");

                let mut graph = Graph::new();
                let session = Session::from_saved_model(
                    &SessionOptions::new(),
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

    loop {
        let (winner,history) = {
            let (ref graph, ref session) = *graph_and_session.lock().unwrap();
            self_play_match(&(|board| evaluator(session, graph, board)))
        };
        for (board, policy) in history.iter() {
            let value = if board.turn() == winner { 1.0 } else { 0.0 };
            fm.append(board, policy, &value);
        }
    }
}
