#![allow(non_snake_case)]

use ggpf::game;
use ggpf::game::breakthrough::*;
use ggpf::game::meta::with_history::*;
use ggpf::game::*;
use ggpf::policies::{get_multi, mcts::muz::*, mcts::puct::*, DynMultiplayerPolicyBuilder};
use ggpf::settings;

use atomic_counter::{AtomicCounter, RelaxedCounter};
use clap::{value_t, App, Arg};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Dim;
use rand::seq::SliceRandom;
use sloth::Lazy;
use std::sync::Arc;
use tokio::runtime;
use typenum::U2;

const MODEL_PATH_ALPHAZERO: &str = "models/alpha-breakthrough/";
const MODEL_PATH_MUZERO: &str = "models/mu-breakthrough/";

type G = WithHistory<Breakthrough, U2>;
//type G = Breakthrough;

pub async fn game_match<
    'a,
    'c,
    'b,
    'd,
    G: game::Game + game::SingleWinner + Clone + 'static,
    GB: game::GameBuilder<G> + Sync,
>(
    n: usize,
    pb1: Box<dyn DynMultiplayerPolicyBuilder<'static, G> + Sync + 'c>,
    pb2: Box<dyn DynMultiplayerPolicyBuilder<'static, G> + Sync + 'd>,
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
    let pb = Arc::new(pb);

    let count_victory_thr: Vec<_> = (0..n)
        .map(|_| {
            let p1 = pb1.create(G::players()[0]);
            let p2 = pb2.create(G::players()[1]);
            let starting_player = *G::players().choose(&mut rand::thread_rng()).unwrap();
            let game = game_factory.create(starting_player);

            let c1 = c1.clone();
            let c2 = c2.clone();
            let pb = pb.clone();

            tokio::spawn(async move {
                let result = if game::simulate(p1, p2, &game).await.last().unwrap().winner()
                    == Some(G::players()[0])
                {
                    c1.inc();
                    1
                } else {
                    c2.inc();
                    0
                };

                if let Some(pb) = pb.as_ref() {
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
        })
        .collect();

    let mut count_victory = 0;
    for thr in count_victory_thr {
        count_victory += thr.await.unwrap();
    }

    if let Some(pb) = pb.as_ref() {
        pb.finish();
    }
    count_victory
}

fn main() {
    let mut threaded_rt = runtime::Builder::new()
        .threaded_scheduler()
        .enable_all()
        .core_threads(8)
        .build()
        .unwrap();

    threaded_rt.block_on(run());
}

async fn run() {
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

    /* Build game to gathe settings*/
    let gb = WithHistoryGB::<_, U2>::new(&BreakthroughBuilder {});
    let g: G = gb.create(G::players()[0]);

    let muz_config = MuZeroConfig {
        action_shape: G::action_dimension(),
        board_shape: g.state_dimension(),
        repr_board_shape: Dim(settings::MUZ_BT_SHAPE),
        batch_size: 1,
        networks_path: MODEL_PATH_MUZERO.into(),
        puct: PUCTSettings {
            DECODE_VALUE: true,
            ..PUCTSettings::default()
        },
        watch_models: false,
    };

    let muz_evals = Lazy::new(|| MuzEvaluators::new(muz_config.clone(), true));

    let alpha_config = AlphaZeroConfig {
        action_shape: G::action_dimension(),
        board_shape: g.state_dimension(),
        batch_size: 1,
        network_path: MODEL_PATH_ALPHAZERO.into(),
        puct: PUCTSettings {
            DECODE_VALUE: false,
            ..PUCTSettings::default()
        },
        watch_models: false,
    };

    let alpha_evals = Lazy::new(|| AlphaZeroEvaluators::new(alpha_config.clone(), true));

    let choice_1 = args.value_of("policy").unwrap_or("rand");
    let p1 = if choice_1 == "alpha" {
        Box::new(PUCT {
            config: alpha_config.puct,
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
            prediction_channel: alpha_evals.get_channel(),
        })
    } else if choice_1 == "mu" {
        Box::new(Muz {
            puct: muz_config.puct,
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
            channels: muz_evals.get_channels(),
            repr_dimension: Dim(settings::MUZ_BT_SHAPE),
        })
    } else {
        get_multi(choice_1)
    };

    //let gb = BreakthroughBuilder {};

    /* Build contender. */
    let choice_2 = args.value_of("against").unwrap_or("rand");
    let p2 = if choice_2 == "alpha" {
        Box::new(PUCT {
            config: alpha_config.puct,
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
            prediction_channel: alpha_evals.get_channel(),
        })
    } else if choice_2 == "mu" {
        Box::new(Muz {
            puct: muz_config.puct,
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
            channels: muz_evals.get_channels(),
            repr_dimension: Dim(settings::MUZ_BT_SHAPE),
        })
    } else {
        get_multi(choice_2)
    };

    let silent = args.is_present("only-result");

    if !silent {
        println!("Player 1: {}", p1);
        println!("Player 2: {}", p2);
    }

    let n_games = value_t!(args.value_of("n"), usize).unwrap_or(100);

    println!("{}", game_match::<G, _>(n_games, p1, p2, &gb, silent).await);
}
