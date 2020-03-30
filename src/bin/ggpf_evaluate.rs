#![allow(non_snake_case)]

use atomic_counter::{AtomicCounter, RelaxedCounter};
use clap::{value_t, App, Arg};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Dim};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

use ggpf::deep::evaluator::{
    dynamics_evaluator_single, prediction_evaluator_single, representation_evaluator_single,
};
use ggpf::deep::tf;
use ggpf::game;
use ggpf::game::breakthrough::*;
use ggpf::game::meta::simulated::Simulated;
use ggpf::game::meta::with_history::*;
use ggpf::policies::{get_multi, mcts::muz::*, mcts::puct::*, DynMultiplayerPolicyBuilder};
use ggpf::settings;

use typenum::U2;

const MODEL_PATH_ALPHAZERO: &str = "models/alpha-breakthrough/";

const MODEL_PATH_MUZERO: &str = "models/mu-breakthrough/";

type G = WithHistory<Breakthrough, U2>;

pub fn game_match<
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

            let result = if game::simulate(p1, p2, &game).last().unwrap().winner()
                == Some(G::players()[0])
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
                .possible_values(&[
                    "rand", "flat", "flat_ucb", "uct", "rave", "ppa", "nmcs", "alpha", "mu",
                ]),
        )
        .arg(
            Arg::with_name("against")
                .short("a")
                .long("against")
                .takes_value(true)
                .possible_values(&[
                    "rand", "flat", "flat_ucb", "uct", "rave", "ppa", "nmcs", "alpha", "mu",
                ]),
        )
        .arg(Arg::with_name("n").short("n").takes_value(true))
        .arg(Arg::with_name("only-result").long("only-result"))
        .get_matches();

    // PUCT
    let alpha_tf = Arc::new(tf::load_model(MODEL_PATH_ALPHAZERO));
    let puct = PUCT {
        _g: PhantomData,
        config: PUCTSettings::default(),
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        evaluate: move |pov, board: &G| {
            let x = alpha_tf.clone();
            let (ref graph, ref session) = x.as_ref();
            prediction_evaluator_single(&session, &graph, pov, board, false)
        },
    };

    // MUZ
    let prediction_path = format!("{}{}", MODEL_PATH_MUZERO, "pv");
    let dynamics_path = format!("{}{}", MODEL_PATH_MUZERO, "dyn");
    let representation_path = format!("{}{}", MODEL_PATH_MUZERO, "state");

    let prediction_tensorflow = Arc::new(tf::load_model(&prediction_path));
    let dynamics_tensorflow = Arc::new(tf::load_model(&dynamics_path));
    let representation_tensorflow = Arc::new(tf::load_model(&representation_path));

    let muz = Muz {
        _h: PhantomData,
        _g: PhantomData,
        puct: PUCTSettings::default(),
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        prediction_evaluate: move |pov: <G as game::Game>::Player, board: &Simulated<G, _, _>| {
            let x = prediction_tensorflow.clone();
            let (ref graph, ref session) = x.as_ref();
            prediction_evaluator_single(&session, &graph, pov, board, true)
        },
        dynamics_evaluate:
            move |board: &Array<f32, _>, action: &Array<f32, <G as game::Feature>::ActionDim>| {
                let x = dynamics_tensorflow.clone();
                let (ref graph, ref session) = x.as_ref();
                dynamics_evaluator_single(
                    &session,
                    &graph,
                    Dim(settings::MUZ_BT_SHAPE),
                    board.clone(),
                    action.clone(),
                    true,
                )
            },
        representation_evaluate: |board: Array<f32, <G as game::Feature>::StateDim>| {
            let x = representation_tensorflow.clone();
            let (ref graph, ref session) = x.as_ref();
            representation_evaluator_single(&session, &graph, Dim(settings::MUZ_BT_SHAPE), board)
        },
    };

    let choice_1 = args.value_of("policy").unwrap_or("rand");
    let p1 = if choice_1 == "alpha" {
        Box::new(puct.clone())
    } else if choice_1 == "mu" {
        Box::new(muz.clone())
    } else {
        get_multi(choice_1)
    };

    /* Build contender. */
    let choice_2 = args.value_of("against").unwrap_or("rand");
    let p2 = if choice_2 == "alpha" {
        Box::new(puct)
    } else if choice_2 == "mu" {
        Box::new(muz)
    } else {
        get_multi(choice_2)
    };

    let gb = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});
    //    let gb = BreakthroughBuilder {};

    let silent = args.is_present("only-result");

    if !silent {
        println!("Player 1: {}", p1);
        println!("Player 2: {}", p2);
    }

    let n_games = value_t!(args.value_of("n"), usize).unwrap_or(100);

    println!("{}", game_match::<_, _>(n_games, p1, p2, &gb, silent));
}
