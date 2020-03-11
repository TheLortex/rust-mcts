#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

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
    get_multi,
    flat::*, mcts::*, nmcs::*, nrpa::*, ppa::*, puct::*, MultiplayerPolicy,
    MultiplayerPolicyBuilder, DynMultiplayerPolicyBuilder, SingleplayerPolicy, SingleplayerPolicyBuilder,
};

fn game_duel<'a, 'b, G: MultiplayerGame>(
    mut p1: Box<dyn MultiplayerPolicy<G> + 'a>,
    mut p2: Box<dyn MultiplayerPolicy<G> + 'b>,
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
    'a, 'c,
    'b, 'd,
    G: MultiplayerGame,
    GB: MultiplayerGameBuilder<G> + Sync,
>(
    n: usize,
    pb1: Box<dyn DynMultiplayerPolicyBuilder<'a, G> + Sync + 'c>, 
    pb2: Box<dyn DynMultiplayerPolicyBuilder<'b, G> + Sync + 'd>,
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

            let p1 = pb1.create(G::players()[0]);
            let p2 = pb2.create(G::players()[1]);

            let starting_player = *G::players().choose(&mut rand::thread_rng()).unwrap();
            let game = game_factory.create(starting_player);

            let result = if game_duel(
                p1,
                p2,
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

use std::collections::HashMap;


fn main() {
    let args = App::new("zerol-evaluate")
        .arg(Arg::with_name("policy")
            .short("p")
            .long("policy")
            .takes_value(true)
            .possible_values(&["rand", "flat", "flat_ucb", "uct", "rave", "ppa", "nmcs"]))
        .get_matches();

    /* Build PUCT */
    let mut graph = Graph::new();
    let session =
        Session::from_saved_model(&SessionOptions::new(), &["serve"], &mut graph, MODEL_PATH)
            .unwrap();

    let puct = PUCT {
        _g: PhantomData,
        C_PUCT: 0.4,
        evaluate: &|board| evaluator(&session, &graph, board),
    };
    let p1 = Box::new(puct);
    
    /* Build contender. */
    let config = args.value_of("policy").unwrap_or("rand");
    let p2 = get_multi(config);

    let gb = BreakthroughBuilder {};

    println!(
        "Result: {}",
        monte_carlo_match::<_, _>(100, p1, p2, &gb)
    );
}
