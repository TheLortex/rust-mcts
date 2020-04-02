#![allow(non_snake_case)]
#![feature(type_alias_impl_trait)]

use ggpf_gym::{gym::*, GymRunnerClient};

use std::future::Future;
use tarpc::{
    client, context,
    server::{BaseChannel, Channel},
};
use tokio::stream::StreamExt;


use std::sync::{Arc, Mutex};
use std::net::{IpAddr, SocketAddr};
use tokio_serde::formats::Json;

use std::time;
use std::thread;

#[tokio::main]
async fn main() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("GYM client!");
    
    let publisher_conn = tarpc::serde_transport::tcp::connect("localhost:1337", Json::default());
    let publisher_conn = publisher_conn.await.unwrap();
    let mut runner =
        GymRunnerClient::new(client::Config::default(), publisher_conn).spawn().unwrap();
    runner.reset(context::current()).await.unwrap();

    thread::sleep_ms(1000);

    let t0 = time::Instant::now();
    let mut i: usize = 0;
    while i < 1000 {
        let res = runner.play(context::current(), (i/4) % 4).await.unwrap();
        //game.render();
        i += 1;

        if res.is_done {
            runner.reset(context::current()).await.unwrap();
        }
    }

    let t1 = time::Instant::now();
    println!("T: {:?}, {}", t1 - t0, i);

}
