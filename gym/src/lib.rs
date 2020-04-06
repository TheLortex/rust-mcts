#![feature(type_alias_impl_trait)]

pub mod gym;

use self::gym::*;

use serde::{Deserialize, Serialize};
use tokio_serde::*;

pub struct BinCodec;

impl<T> Serializer<T> for BinCodec
where
    T: Serialize,
{
    type Error = std::io::Error;

    fn serialize(self: std::pin::Pin<&mut Self>, val: &T) -> Result<bytes::Bytes, Self::Error> {
        bincode::serialize(val)
            .map(|x| x.into())
            .map_err(|x| match *x {
                bincode::ErrorKind::Io(io) => io,
                _ => std::io::Error::new(std::io::ErrorKind::Other, "non-io error."),
            })
    }
}

impl<T> Deserializer<T> for BinCodec
where
    T: for<'de> Deserialize<'de>,
{
    type Error = std::io::Error;

    fn deserialize(
        self: std::pin::Pin<&mut Self>,
        bytes: &bytes::BytesMut,
    ) -> Result<T, Self::Error> {
        bincode::deserialize(bytes).map_err(|x| match *x {
            bincode::ErrorKind::Io(io) => io,
            _ => std::io::Error::new(std::io::ErrorKind::Other, "non-io error."),
        })
    }
}

impl Default for BinCodec {
    fn default() -> Self {
        Self {}
    }
}

#[tarpc::service]
pub trait GymRunner {
    async fn init(name: String, render: bool);
    async fn reset() -> SpaceData;
    async fn play(action: usize) -> State;
    async fn action_space() -> SpaceTemplate;
    async fn observation_space() -> SpaceTemplate;
}
