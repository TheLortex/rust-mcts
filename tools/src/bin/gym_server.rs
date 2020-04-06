#![allow(non_snake_case)]
#![feature(type_alias_impl_trait)]

use ggpf_gym::*;

use std::future::Future;
use tarpc::{
    context,
    server::{BaseChannel, Channel},
};
use tokio::stream::StreamExt;

use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct GymServer {
    game: Arc<Mutex<Option<(gym::Environment, bool)>>>,
}

#[doc(ignore)]
impl GymRunner for GymServer {
    type InitFut = impl Future<Output = ()> + Send;

    fn init(self, _: context::Context, game: String, render: bool) -> Self::InitFut {
        log::info!("Init.");
        async move {
            let gym = gym::GymClient::default();
            let game = gym.make(&game);

            *self.game.lock().unwrap() = Some((game, render));
        }
    }

    type ResetFut = impl Future<Output = gym::SpaceData>;

    fn reset(self, _: context::Context) -> Self::ResetFut {
        log::info!("Reset.");

        async move {
            if let Some((ref game, render)) = *self.game.lock().unwrap() {
                let res = game.reset().unwrap();
                if render {
                    game.render();
                }
                res
            } else {
                panic!("The game hasn't been initialized.");
            }
        }
    }

    type PlayFut = impl Future<Output = gym::State>;

    fn play(self, _: context::Context, action: usize) -> Self::PlayFut {
        log::info!("Play {}.", action);

        async move {
            if let Some((ref game, render)) = *self.game.lock().unwrap() {
                let res = game.step(&gym::SpaceData::DISCRETE(action)).unwrap();
                if render {
                    game.render();
                }
                res
            } else {
                panic!("The game hasn't been initialized.");
            }
        }
    }

    type ActionSpaceFut = impl Future<Output = gym::SpaceTemplate>;

    fn action_space(self, _: context::Context) -> Self::ActionSpaceFut {
        log::info!("Action space");
        async move {
            if let Some((ref game, _)) = *self.game.lock().unwrap() {
                game.action_space().clone()
            } else {
                panic!("The game hasn't been initialized.");
            }
        }
    }

    type ObservationSpaceFut = impl Future<Output = gym::SpaceTemplate>;

    fn observation_space(self, _: context::Context) -> Self::ObservationSpaceFut {
        log::info!("Observation space");
        async move {
            if let Some((ref game, _)) = *self.game.lock().unwrap() {
                game.observation_space().clone()
            } else {
                panic!("The game hasn't been initialized.");
            }
        }
    }
}

#[tokio::main]
async fn main() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("GYM!");
    let transport = tarpc::serde_transport::tcp::listen("localhost:1337", BinCodec::default)
        .await
        .unwrap();
    let addr = transport.local_addr();
    log::info!("Listening on {}", addr);

    let mut stream = transport
        .filter_map(|r| r.ok())
        .map(BaseChannel::with_defaults);

    tokio::spawn(async move {
        while let Some(client) = stream.next().await {
            tokio::spawn(async {
                log::info!("New client.");
                let game = Arc::new(Mutex::new(None));
                let gym_server = GymServer { game: game.clone() };
                client.respond_with(gym_server.serve()).execute().await;
                log::info!("Disconnected.");

                if let Some((ref game, _)) = *game.lock().unwrap() {
                    game.close();
                };
            });
        }
    })
    .await
    .unwrap();
}
