
#![feature(type_alias_impl_trait)]

pub mod gym;

use self::gym::*;

use std::future::Future;
use tarpc::{
    client, context,
    server::{BaseChannel, Channel},
};
use tokio::stream::StreamExt;

#[tarpc::service]
pub trait GymRunner {
    async fn reset();
    async fn play(action: usize) -> State;
}