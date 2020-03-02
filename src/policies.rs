use super::game::Game;

pub mod flat;
pub mod mcts;
pub mod nmcs;
pub mod nrpa;
pub mod ppa;

pub const N_PLAYOUTS: usize = 1000;

/* POLICY TRAITS */

pub trait PolicyBuilder<T: Game> {
    type P: Policy<T>;

    fn create(&self, color: T::Player) -> Self::P; 
}

pub trait Policy<T: Game> {
    fn play(&mut self, board: &T) -> T::Move;
}

pub trait SinglePolicyBuilder<T: Game> {
    type P: SinglePolicy<T>;

    fn create(&self, color: T::Player) -> Self::P; 
}

pub trait SinglePolicy<T: Game> {
    fn solve(&mut self, board: &T) -> Vec<T::Move>;
}