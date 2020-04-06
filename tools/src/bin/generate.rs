#![allow(non_snake_case)]

//! # GENERATE - self-play games generator.
//!
//! Usage: `cargo run --release --bin generate -- -c breakthrough -m alpha|mu`
//!
//! Generates games continuously using latest network in the corresponding directory.
//! This supposes that the network has been created and that the trainer is running, consuming
//! generated games.
//! To launch the trainer, use `python training.py --config breakthrough -m alpha|mu`.

use ggpf::deep::file_manager;
use ggpf::deep::self_play::GameHistoryEntry;
use ggpf::game::breakthrough::BreakthroughBuilder;
use ggpf::game::meta::with_history::*;
use ggpf::game::openai::GymBuilder;
use ggpf::game::*;
use ggpf::settings::{self, Config, Method, StrError};

use std::fs;
use tokio::runtime;
use tokio::sync::mpsc;

use clap::{App, Arg};

use std::error;

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
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("AlphaZero generate: starting!");

    let args = App::new("ggpf-generate")
        .arg(
            Arg::with_name("method")
                .short("m")
                .long("method")
                .takes_value(true)
                .possible_values(&["alpha", "mu"]),
        )
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .takes_value(true),
        )
        .get_matches();

    let config_file = format!("config/{}.toml", args.value_of("config").unwrap());
    let config = fs::read_to_string(config_file)?;

    let config: Config = toml::from_str(&config)?;

    let method: Method = match args.value_of("method").unwrap() {
        "alpha" => Method::AlphaZero,
        "mu" => Method::MuZero,
        _ => panic!("Unknown method"),
    };

    match config.game.clone() {
        settings::Game::Breakthrough { size, history } => {
            let gb = BreakthroughBuilder { size };
            if let Some(history) = history {
                let game_builder = WithHistoryGB::new(gb, history);
                run_generator(config, game_builder, method).await
            } else {
                let game_builder = gb;
                run_generator(config, game_builder, method).await
            }
        }
        settings::Game::Gym {
            name,
            remote,
            history,
        } => {
            let gb = GymBuilder {
                address: remote,
                game_name: name,
                render: false,
            };

            if let Some(history) = history {
                let game_builder = WithHistoryGB::new(gb, history);
                run_generator(config, game_builder, method).await
            } else {
                let game_builder = gb;
                run_generator(config, game_builder, method).await
            }
        }
    }?;
    Ok(())
}

async fn run_generator<GB: GameBuilder>(config: Config, gb: GB, method: Method) -> Result<()>
where
    GB: GameBuilder + Clone + Send + Sync + 'static,
    GB::G: Features + Clone,
{
    let state: GB::G = gb.create(<GB::G as Game>::players()[0]).await;
    let ft = state.get_features();

    let board_shape = <GB::G as Features>::state_dimension(&ft);
    let action_shape = <GB::G as Features>::action_dimension(&ft);

    log::debug!("Action: {:?}", action_shape);
    log::debug!("Board: {:?}", board_shape);

    // Game channel.
    let (tx_games, mut rx_games) = mpsc::channel::<GameHistoryEntry<GB::G>>(1024);

    match method {
        Method::AlphaZero => {
            if let Some(alpha_config) = config.get_alphazero(action_shape, board_shape) {
                tokio::spawn(ggpf::deep::self_play::alphazero_game_generator(
                    alpha_config,
                    config.self_play,
                    gb,
                    tx_games,
                ));
            } else {
                return Err(Box::new(StrError(
                    "Alpha is not supported for this game.".to_owned(),
                )));
            }
        }
        Method::MuZero => {
            if let Some(mu_config) = config.get_muzero(action_shape, board_shape) {
                tokio::spawn(ggpf::deep::self_play::muzero_game_generator(
                    mu_config,
                    config.self_play,
                    gb,
                    tx_games,
                ));
            } else {
                return Err(Box::new(StrError(
                    "Mu is not supported for this game.".to_owned(),
                )));
            }
        }
    }

    // Game writer.
    let mut fm = file_manager::FileManager::new(&format!(
        "./data/{}-{}/fifo",
        method.name(),
        config.game.name()
    ));
    let game_writer = tokio::spawn(async move {
        while let Some(game) = rx_games.recv().await {
            fm.append(game);
        }
    });

    game_writer.await?;
    Ok(())
}
