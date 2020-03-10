use super::game::{SingleplayerGame, MultiplayerGame};

pub mod flat;
pub mod mcts;
pub mod nmcs;
pub mod nrpa;
pub mod ppa;
pub mod puct;

pub const N_PLAYOUTS: usize = 100;

/* POLICY TRAITS */

pub trait MultiplayerPolicyBuilder<T: MultiplayerGame> {
    type P: MultiplayerPolicy<T>;

    fn create(&self, color: T::Player) -> Self::P; 
}

pub trait MultiplayerPolicy<T: MultiplayerGame> {
    fn play(&mut self, board: &T) -> T::Move;
}

pub trait SingleplayerPolicyBuilder<T: SingleplayerGame> {
    type P: SingleplayerPolicy<T>;

    fn create(&self) -> Self::P; 
}

pub trait SingleplayerPolicy<T: SingleplayerGame> {
    fn solve(&mut self, board: &T) -> Vec<T::Move>;
}