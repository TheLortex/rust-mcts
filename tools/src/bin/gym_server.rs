#![allow(non_snake_case)]
#![feature(type_alias_impl_trait)]

use ggpf_gym::gym::*;

use std::future::Future;
use tarpc::{
    client, context,
    server::{BaseChannel, Channel},
};
use tokio::stream::StreamExt;

#[tarpc::service]
trait GymRunner {
    async fn reset();
    async fn play(action: usize) -> State;
}

use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct GymServer {
    game: Arc<Mutex<Environment>>,
}

impl GymRunner for GymServer {
    type ResetFut = impl Future<Output = ()>;

    fn reset(self, _: context::Context) -> Self::ResetFut {
        log::debug!("Reset.");
    
        async move {
            let game = self.game.lock().unwrap();
            game.reset().unwrap();
            game.render();
        }
    }

    type PlayFut = impl Future<Output = State>;

    fn play(self, _: context::Context, action: usize) -> Self::PlayFut {
        log::debug!("Play {}.", action);

        async move { 
            let game = self.game.lock().unwrap();
            let res = game.step(&SpaceData::DISCRETE(action)).unwrap();
            game.render();
            res
         }
    }
}

use std::net::{IpAddr, SocketAddr};
use tokio_serde::formats::Json;

#[tokio::main]
async fn main() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("GYM!");
    let mut transport = tarpc::serde_transport::tcp::listen("localhost:1337", Json::default)
        .await
        .unwrap();
    let addr = transport.local_addr();
    log::info!("Listening on {}", addr);

    // For this example, we're just going to wait for one connection.
    let client = transport.next().await.unwrap().unwrap();

    let gym = GymClient::default();
    let game = gym.make("CarRacing-v0");

    let gym_server = GymServer {
        game: Arc::new(Mutex::new(game))
    };

    // `Channel` is a trait representing a server-side connection. It is a trait to allow
    // for some channels to be instrumented: for example, to track the number of open connections.
    // BaseChannel is the most basic channel, simply wrapping a transport with no added
    // functionality.
    BaseChannel::with_defaults(client)
        // serve_world is generated by the tarpc::service attribute. It takes as input any type
        // implementing the generated World trait.
        .respond_with(gym_server.serve())
        .execute()
        .await;

    /*
    let gym = GymClient::default();
    let game = gym.make("MsPacman-v0");
    log::info!("{:?}", game.reset());

    let t0 = time::Instant::now();
    let mut i = 0;
    while i < 1000 {
        let res = game.step(&SpaceData::DISCRETE(0)).unwrap();
        //game.render();
        i += 1;

        if res.is_done {
            game.reset();
        }
    }

    let t1 = time::Instant::now();
    println!("T: {:?}, {}", t1 - t0, i);*/

    /*
    let mut threaded_rt = runtime::Builder::new()
        .threaded_scheduler()
        .enable_all()
        .core_threads(8)
        .build()
        .unwrap();

    threaded_rt.block_on(run());*/
}
