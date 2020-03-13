use super::game::{MultiplayerGame, SingleplayerGame};

pub mod flat;
pub mod mcts;
pub mod nmcs;
pub mod nrpa;
pub mod ppa;
pub mod puct;
pub mod puct_async; 
pub mod mcts_async; 

use async_trait::async_trait;

pub const N_PLAYOUTS: usize = 100;

use std::fmt::Display;
/* MULTIPLAYER POLICY TRAITS */
/**
 * A static policy.
 */
pub trait MultiplayerPolicy<T: MultiplayerGame> {
    fn play(&mut self, board: &T) -> T::Move;
}
#[async_trait]
pub trait AsyncMultiplayerPolicy<T: MultiplayerGame> {
    async fn play(&mut self, board: &T) -> T::Move;
}
/**
 * A static policy builder.
 */
pub trait MultiplayerPolicyBuilder<T: MultiplayerGame>: Display {
    type P: MultiplayerPolicy<T>;

    fn create(&self, color: T::Player) -> Self::P;
}
pub trait AsyncMultiplayerPolicyBuilder<T: MultiplayerGame>: Display {
    type P: AsyncMultiplayerPolicy<T>;

    fn create(&self, color: T::Player) -> Self::P;
}
/**
 * A dynamic policy builder.
 */
pub trait DynMultiplayerPolicyBuilder<'a, T: MultiplayerGame>: Display {
    fn create(&self, color: T::Player) -> Box<dyn MultiplayerPolicy<T> + 'a>;
}

impl<'a, G, PB> DynMultiplayerPolicyBuilder<'a, G> for PB
where
    G: MultiplayerGame,
    PB: MultiplayerPolicyBuilder<G>,
    PB::P: 'a,
{
    fn create(&self, color: G::Player) -> Box<dyn MultiplayerPolicy<G> + 'a> {
        Box::new(PB::create(self, color))
    }
}

/* SINGLEPLAYER POLICY TRAITS */

pub trait SingleplayerPolicyBuilder<T: SingleplayerGame> {
    type P: SingleplayerPolicy<T>;

    fn create(&self) -> Self::P;
}

pub trait SingleplayerPolicy<T: SingleplayerGame> {
    fn solve(&mut self, board: &T) -> Vec<T::Move>;
}

/** MULTIPLAYER POLICIES */

use super::game::NoFeatures;

pub fn get_multi<'a, G: MultiplayerGame + 'a>(name: &str) -> Box<dyn DynMultiplayerPolicyBuilder<'a, G> + Sync + 'a> {
    match name {
        "rand" => Box::new(flat::Random::default()),
        "flat" => Box::new(flat::FlatMonteCarlo::default()),
        "flat_ucb" => Box::new(flat::FlatUCBMonteCarlo::default()),
        "uct" => Box::new(mcts::UCT::default()),
        "rave" => Box::new(mcts::RAVE::default()),
        "ppa" => Box::new(ppa::PPA::<_, NoFeatures>::default()),
        "nmcs" => Box::new(nmcs::MultiNMCS::default()),
        _ => panic!("Policy '{}' not found.", name)
    }
}

/* SINGLEPLAYER POLICIES: TODO */
