#![allow(non_snake_case)]

use atomic_counter::{AtomicCounter, RelaxedCounter};

use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::hash::Hasher;

use rand::seq::SliceRandom;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

extern crate zerol;

use zerol::game;
use zerol::game::breakthrough::*;
use zerol::game::{MoveTrait, MoveCode, MultiplayerGame, MultiplayerGameBuilder, SingleplayerGame};
use zerol::policies::{
    flat::*, mcts::puct::*, MultiplayerPolicyBuilder, SingleplayerPolicy,
    SingleplayerPolicyBuilder,
};
use zerol::settings;

fn game_solo<G: SingleplayerGame, P: SingleplayerPolicyBuilder<G>>(pb: &P, game: &G) -> f32 {
    let mut p = pb.create();
    let mut b = game.clone();
    let actions = p.solve(&b);
    for a in actions {
        println!("{:?}", b);
        println!("Action: {:?}", a);
        b.play(&a);
        //   println!("Score: {}", b.score(player));
    }
    println!("{:?}", b);
    b.score()
}

pub fn monte_carlo_match<
    G: MultiplayerGame,
    GB: MultiplayerGameBuilder<G> + Sync,
    P1: MultiplayerPolicyBuilder<G> + Sync,
    P2: MultiplayerPolicyBuilder<G> + Sync,
>(
    n: usize,
    pb1: &P1,
    pb2: &P2,
    game_factory: &GB,
) -> usize {
    let pb = ProgressBar::new(n as u64);
    pb.set_style(ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg} (ETA {eta})",
    ));
    pb.enable_steady_tick(200);

    let c1 = Arc::new(RelaxedCounter::new(0));
    let c2 = Arc::new(RelaxedCounter::new(0));

    let count_victory: usize = (0..n)
        .into_par_iter()
        .map(|_| {
            let c1 = c1.clone();
            let c2 = c2.clone();

            let p1 = Box::new(pb1.create(G::players()[0]));
            let p2 = Box::new(pb2.create(G::players()[1]));

            let starting_player = *G::players().choose(&mut rand::thread_rng()).unwrap();
            let game = game_factory.create(starting_player);

            let result = if game::simulate(p1, p2, &game)
                .last()
                .unwrap()
                .has_won(G::players()[0])
            {
                c1.inc();
                1
            } else {
                c2.inc();
                0
            };
            pb.inc(1);
            let v1 = c1.get();
            let v2 = c2.get();
            pb.set_message(&format!(
                "{}/{} ({:.2}%)",
                v1,
                v2,
                (v1 as f32) * 100. / ((v1 + v2) as f32)
            ));
            result
        })
        .sum();
    pb.finish();
    count_victory
}

pub struct BTCapture {}
impl MoveCode<Breakthrough> for BTCapture {
    fn code(game: &Breakthrough, action: &zerol::game::breakthrough::Move) -> usize {
        let mut s = DefaultHasher::new();
        action.hash(&mut s);
        let capture = {
            let (tx, ty) = action.target(&game.content);
            if game.content[tx][ty] == zerol::game::breakthrough::Cell::Empty {
                0
            } else {
                1
            }
        };
        capture.hash(&mut s);
        usize::try_from(s.finish()).unwrap()
    }
}

use std::marker::PhantomData;
use zerol::misc::breakthrough_evaluator;

use tensorflow::{Graph, Session, SessionOptions};
const MODEL_PATH: &str = "models/sample";

fn main() {
    let mut graph = Graph::new();
    let session =
        Session::from_saved_model(&SessionOptions::new(), &["serve"], &mut graph, MODEL_PATH)
            .unwrap();

    let p1 = PUCT {
        _g: PhantomData,
        C_PUCT: 0.4,
        N_HISTORY: settings::DEFAULT_N_HISTORY_PUCT,
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        evaluate: &(|pov, board_history: &[Breakthrough]| {
            breakthrough_evaluator(&session, &graph, pov, board_history)
        }),
    };
    let p2 = FlatMonteCarlo::default();

    let gb = BreakthroughBuilder {};

    println!(
        "Result: {}",
        monte_carlo_match::<_, _, _, _>(100, &p1, &p2, &gb)
    );
    //main_ui();
    /*
    let pb = NRPA::<_, NoFeatures>::default();
    let res = game_solo::<WeakSchurNumber, _>(&pb, ());
    println!("=> {} ", res);*/
    /*
    let pb = NRPA::<_, NoFeatures>::default();
    let config = Hashcode20Settings::new_from_file("./data/b_read_on.txt");
    let res = game_solo::<Hashcode20, _>(&pb, &config);
    println!("=> {} ", res);*/
}
