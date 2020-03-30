#![allow(non_snake_case)]

use atomic_counter::{AtomicCounter, RelaxedCounter};
use clap::{App, Arg, value_t_or_exit};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use tensorflow::{Graph, Session, SessionOptions};
use ggpf::game;
use ggpf::game::breakthrough::*;
use ggpf::game::meta::with_history::*;
use ggpf::misc::tf::game_evaluator;
use ggpf::policies::{get_multi, mcts::puct::*, DynMultiplayerPolicyBuilder};
use ggpf::settings;

use typenum::U2;

const MODEL_PATH: &str = "models/breakthrough";

pub fn monte_carlo_match<
    'a,
    'c,
    'b,
    'd,
    G: game::Game + game::SingleWinner + Clone,
    GB: game::GameBuilder<G> + Sync,
>(
    n: usize,
    pb1: Box<dyn DynMultiplayerPolicyBuilder<'a, G> + Sync + 'c>,
    pb2: Box<dyn DynMultiplayerPolicyBuilder<'b, G> + Sync + 'd>,
    game_factory: &GB,
    silent: bool,
) -> usize {
    let pb = if silent {
        None
    } else {
        let pb = ProgressBar::new(n as u64);
        pb.set_style(ProgressStyle::default_bar().template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg} (ETA {eta})",
        ));
        pb.enable_steady_tick(200);
        Some(pb)
    };

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

            let result = if game::simulate(p1, p2, &game)
                .last()
                .unwrap()
                .winner() == Some(G::players()[0])
            {
                c1.inc();
                1
            } else {
                c2.inc();
                0
            };

            if let Some(pb) = &pb {
                pb.inc(1);
                let v1 = c1.get();
                let v2 = c2.get();
                pb.set_message(&format!(
                    "{}/{} ({:.2}%)",
                    v1,
                    v2,
                    (v1 as f32) * 100. / ((v1 + v2) as f32)
                ));
            }
            result
        })
        .sum();

    if let Some(pb) = pb {
        pb.finish();
    }
    count_victory
}

fn main() {
    let args = App::new("zerol-evaluate")
        .arg(
            Arg::with_name("policy")
                .short("p")
                .long("policy")
                .takes_value(true)
                .possible_values(&["rand", "flat", "flat_ucb", "uct", "rave", "ppa", "nmcs"]),
        )
        .arg(
            Arg::with_name("against")
                .short("a")
                .long("against")
                .takes_value(true)
                .possible_values(&["rand", "flat", "flat_ucb", "uct", "rave", "ppa", "nmcs"]),
        )
        .arg(
            Arg::with_name("n")
                .short("n")
                .takes_value(true)
        )
        .arg(Arg::with_name("only-result").long("only-result"))
        .get_matches();

    /* Build PUCT */
    let mut graph = Graph::new();
    let session =
        Session::from_saved_model(&SessionOptions::new(), &["serve"], &mut graph, MODEL_PATH)
            .unwrap();

    let puct = PUCT {
        _g: PhantomData,
        config: PUCTSettings::default(),
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        evaluate: |pov, board: &WithHistory<Breakthrough, U2>| {
            //evaluate: |pov, board: &Breakthrough| {
            game_evaluator(&session, &graph, pov, board)
        },
    };
    
    let p1 = if let Some(val) = args.value_of("against") {
        get_multi(val)
    } else {
        
        Box::new(puct)
    };
    /* Build contender. */
    let config = args.value_of("policy").unwrap_or("rand");
    let p2 = get_multi(config);

    let gb = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});
    //    let gb = BreakthroughBuilder {};

    let silent = args.is_present("only-result");

    if !silent {
        println!("Player 1: {}", p1);
        println!("Player 2: {}", p2);
    }

    let n_games = value_t_or_exit!(args.value_of("n"), usize);

    println!("{}", monte_carlo_match::<_, _>(n_games, p1, p2, &gb, silent));
}
