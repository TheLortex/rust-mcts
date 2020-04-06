#![allow(non_snake_case)]

//! # EVALUATE - compare policies on a two-player game.
//!
//! Usage: `cargo run --release --bin evaluate -- -c breakthrough -p ppa -a puct -n 100`
//!
//! Launches `-n` games with a random starting player and count victories for the first policy.

use ggpf::game;
use ggpf::game::breakthrough::*;
use ggpf::game::meta::with_history::*;
use ggpf::game::*;
use ggpf::policies::{get_multi, mcts::muz::*, mcts::puct::*, DynMultiplayerPolicyBuilder};
use ggpf::settings::{self, Config, StrError};

use atomic_counter::{AtomicCounter, RelaxedCounter};
use clap::{value_t, App, Arg};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use sloth::Lazy;
use std::error;
use std::fs;
use std::sync::Arc;
use tokio::runtime;

pub async fn game_match<'a, 'c, 'b, 'd, GB>(
    n: usize,
    pb1: Box<dyn DynMultiplayerPolicyBuilder<'static, GB::G> + Sync + 'c>,
    pb2: Box<dyn DynMultiplayerPolicyBuilder<'static, GB::G> + Sync + 'd>,
    game_factory: GB,
    silent: bool,
) -> usize
where
    GB::G: game::Game + game::SingleWinner + 'static,
    GB: game::GameBuilder + Clone + Sync + Send + 'static,
{
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
            let p1 = pb1.create(<GB::G as Game>::players()[0]);
            let p2 = pb2.create(<GB::G as Game>::players()[1]);
            let starting_player = *<GB::G as Game>::players()
                .choose(&mut rand::thread_rng())
                .unwrap();

            let c1 = c1.clone();
            let c2 = c2.clone();
            let pb = pb.clone();
            let game_factory = game_factory.clone();

            tokio::spawn(async move {
                let mut game = game_factory.create(starting_player).await;

                game::simulate(p1, p2, &mut game).await;

                let result = if game.winner() == Some(<GB::G as Game>::players()[0]) {
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

type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

fn main() {
    let mut threaded_rt = runtime::Builder::new()
        .threaded_scheduler()
        .enable_all()
        .core_threads(8)
        .build()
        .unwrap();

    if let Err(e) = threaded_rt.block_on(run()) {
        println!("Error: {}", e)
    }
}

async fn run() -> Result<()> {
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
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .takes_value(true)
                .possible_values(&["breakthrough", "atari"]),
        )
        .arg(Arg::with_name("n").short("n").takes_value(true))
        .arg(Arg::with_name("only-result").long("only-result"))
        .get_matches();

    let config_file = format!("config/{}.toml", args.value_of("config").unwrap());
    let config = fs::read_to_string(config_file)?;

    let config: Config = toml::from_str(&config)?;

    match config.game {
        settings::Game::Breakthrough { size, history } => {
            if let Some(history) = history {
                let game_builder = WithHistoryGB::new(BreakthroughBuilder { size }, history);
                next(config, args, game_builder).await
            } else {
                let game_builder = BreakthroughBuilder { size };
                next(config, args, game_builder).await
            }
        }
        settings::Game::Gym { .. } => {
            return Err(Box::new(StrError(
                "Gym has not been implemented yet.".to_owned(),
            )))
        }
    }?;
    Ok(())
}

use std::hash::Hash;

async fn next<GB>(config: Config, args: clap::ArgMatches<'_>, game_builder: GB) -> Result<()>
where
    GB: GameBuilder + 'static,
    GB::G: Features + SingleWinner + Clone + Hash + Eq + 'static,
{
    /* Build game to gathe settings*/
    let g: GB::G = game_builder.create(<GB::G as Game>::players()[0]).await;
    let ft = g.get_features();

    let board_shape = <GB::G as Features>::state_dimension(&ft);
    let action_shape = <GB::G as Features>::action_dimension(&ft);

    let muz_config = config
        .get_muzero(action_shape.clone(), board_shape.clone())
        .map(|mut x| {
            x.watch_models = false;
            x.batch_size = 50;
            x
        });
    let muz_evals = Lazy::new(|| MuzEvaluators::new(muz_config.unwrap(), true));

    let alpha_config = config
        .get_alphazero(action_shape, board_shape)
        .map(|mut x| {
            x.watch_models = false;
            x.batch_size = 50;
            x
        });

    let alpha_evals = Lazy::new(|| AlphaZeroEvaluators::new(alpha_config.unwrap(), true));

    let choice_1 = args.value_of("policy").unwrap_or("rand");
    let p1 = if choice_1 == "alpha" {
        let alpha_conf = config.alpha.expect("Alpha not configured.");
        Box::new(PUCT {
            config: alpha_conf.puct,
            n_playouts: config.mcts.playouts,
            prediction_channel: alpha_evals.get_channel(),
        })
    } else if choice_1 == "mu" {
        let mu_conf = config.mu.expect("Mu not configured.");
        Box::new(Muz {
            muz: mu_conf,
            n_playouts: config.mcts.playouts,
            channels: muz_evals.get_channels(),
        })
    } else {
        get_multi(config.clone(), choice_1)
    };

    //let gb = BreakthroughBuilder {};

    /* Build contender. */
    let choice_2 = args.value_of("against").unwrap_or("rand");
    let p2 = if choice_2 == "alpha" {
        let alpha_conf = config.alpha.expect("Alpha not configured.");
        Box::new(PUCT {
            config: alpha_conf.puct,
            n_playouts: config.mcts.playouts,
            prediction_channel: alpha_evals.get_channel(),
        })
    } else if choice_2 == "mu" {
        let mu_conf = config.mu.expect("Mu not configured.");
        Box::new(Muz {
            muz: mu_conf,
            n_playouts: config.mcts.playouts,
            channels: muz_evals.get_channels(),
        })
    } else {
        get_multi(config, choice_2)
    };

    let silent = args.is_present("only-result");

    if !silent {
        println!("Player 1: {}", p1);
        println!("Player 2: {}", p2);
    }

    let n_games = value_t!(args.value_of("n"), usize).unwrap_or(100);

    println!(
        "{}",
        game_match(n_games, p1, p2, game_builder, silent).await
    );
    Ok(())
}
