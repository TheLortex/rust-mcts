pub mod mcts;
pub mod puct;

use crate::game::MultiplayerGame;
use std::fmt::Display;

use async_trait::async_trait;

#[async_trait]
pub trait AsyncMultiplayerPolicy<T: MultiplayerGame> {
    async fn play(&mut self, board: &T) -> T::Move;
}


pub trait AsyncMultiplayerPolicyBuilder<T: MultiplayerGame>: Display {
    type P: AsyncMultiplayerPolicy<T>;

    fn create(&self, color: T::Player) -> Self::P;
}