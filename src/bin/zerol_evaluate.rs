#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![feature(impl_trait_in_bindings)]

use cursive::views::{Button, Dialog, LinearLayout, NamedView};
use cursive::Cursive;

use atomic_counter::{AtomicCounter, RelaxedCounter};

use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

use rand::seq::SliceRandom;
use rand::Rng;
use std::cell::RefCell;
use std::sync::Arc;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

extern crate zerol;

use zerol::game::breakthrough::*;
use zerol::game::hashcode_20::*;
use zerol::game::misere_breakthrough::*;
use zerol::game::weak_schur::*;
use zerol::game::*;
use zerol::policies::{
    flat::*, mcts::*, nmcs::*, nrpa::*, ppa::*, puct::*, MultiplayerPolicy,
    MultiplayerPolicyBuilder, SingleplayerPolicy, SingleplayerPolicyBuilder,
};

fn game_duel<G: MultiplayerGame, P1: MultiplayerPolicy<G>, P2: MultiplayerPolicy<G>>(
    p1: &mut P1,
    p2: &mut P2,
    game: &G,
) -> G {
    let mut b = game.clone();
    const DBG: bool = false;

    while {
        let action = if b.turn() == G::players()[0] {
            p1.play(&b)
        } else {
            p2.play(&b)
        };
        if DBG {
            println!("{:?} => {:?}", b, action);
        }
        b.play(&action);
        !b.is_finished()
    } {}
    if DBG {
        println!("{:?}", b);
    };
    b
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
    pb.set_style(
        ProgressStyle::default_bar().template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg} (ETA {eta})",
        ),
    );
    pb.enable_steady_tick(200);

    let c1 = Arc::new(RelaxedCounter::new(0));
    let c2 = Arc::new(RelaxedCounter::new(0));

    let count_victory: usize = (0..n)
        .into_par_iter()
        .map(|_| {
            let c1 = c1.clone();
            let c2 = c2.clone();

            let mut p1 = pb1.create(G::players()[0]);
            let mut p2 = pb2.create(G::players()[1]);

            let starting_player = *G::players().choose(&mut rand::thread_rng()).unwrap();
            let game = game_factory.create(starting_player);

            let result = if game_duel(
                &mut p1,
                &mut p2,
                &game,
            ).has_won(G::players()[0])
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

type G = Breakthrough;


use std::marker::PhantomData;
use zerol::misc::evaluator;

use tensorflow::{Code, Graph, Session, SessionOptions, Status};
const MODEL_PATH: &str = "models/sample";

use clap::{Arg, App, SubCommand};

fn main() {
    let matches = App::new("zerol-evaluate")
        .arg(Arg::with_name("policy")
            .short("p")
            .long("policy")
            .possible_values(&["rand", "flat", "flat_ucb", "uct", "rave", "ppa", "nmcs"]))
        .get_matches();

    let config = matches.value_of("policy").unwrap_or("rand");

    let rand: Box<&dyn MultiplayerPolicyBuilder<G>> = Box::new(&Random::default() as &dyn MultiplayerPolicyBuilder<G>);
/*
    let p2: Box<&dyn MultiplayerPolicyBuilder<G>> = match config {
        "rand" => Box::new(),
       /* "flat" => Box::new(FlatMonteCarlo::default() as dyn MultiplayerPolicyBuilder<G>),
        "flat_ucb" => Box::new(FlatUCBMonteCarlo::default() as dyn MultiplayerPolicyBuilder<G>),
        "uct" => Box::new(UCT::default() as dyn MultiplayerPolicyBuilder<G>),
        "rave" => Box::new(RAVE::default() as dyn MultiplayerPolicyBuilder<G>),
        "ppa" => Box::new(PPA::default() as dyn MultiplayerPolicyBuilder<G>),
        "nmcs" => Box::new(MultiNMCS::default() as dyn MultiplayerPolicyBuilder<G>)*/
    };*/
 

    let mut graph = Graph::new();
    let session =
        Session::from_saved_model(&SessionOptions::new(), &["serve"], &mut graph, MODEL_PATH)
            .unwrap();

    let p1 = PUCT {
        _g: PhantomData,
        C_PUCT: 0.4,
        evaluate: &(|board| evaluator(&session, &graph, board)),
    };

    let gb = BreakthroughBuilder {};

    println!(
        "Result: {}",
        monte_carlo_match::<_, _, _, _>(100, &p1, &p2, &gb)
    );
}
