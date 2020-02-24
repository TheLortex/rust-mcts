use super::game::Game;

pub mod flat;
pub mod mcts;
pub mod nmcs;
pub mod nrpa;
pub mod ppa;

pub const N_PLAYOUTS: usize = 100;

/* POLICY TRAITS */

pub trait PolicyBuilder<T: Game> {
    type P: Policy<T>;

    fn create(&self, color: T::Player) -> Self::P; 
}

pub trait Policy<T: Game> {
    fn play(&mut self, board: &T) -> T::Move;
}
