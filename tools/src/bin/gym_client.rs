#![allow(non_snake_case)]
#![feature(type_alias_impl_trait)]

use ggpf_gym::{GymRunnerClient, BinCodec};

use tarpc::{
    client, context,
};

use std::thread;
use std::time;


#[tokio::main]
async fn main() {
    flexi_logger::Logger::with_env().start().unwrap();
    log::info!("GYM client!");
    
    let publisher_conn = tarpc::serde_transport::tcp::connect("localhost:1337", BinCodec::default());
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
