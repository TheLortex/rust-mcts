use crate::game::{Game};

use std::fmt::Display;
use async_trait::async_trait;

pub mod flat;
pub mod mcts;
pub mod nmcs;
pub mod nrpa;
pub mod ppa;

/* MULTIPLAYER POLICY TRAITS */

/**
 * A static policy.
 */

pub trait MultiplayerPolicy<T: Game> {
    fn play(&mut self, board: &T) -> T::Move;
}
/**
 * A static policy builder.
 */
pub trait MultiplayerPolicyBuilder<T: Game>: Display {
    type P: MultiplayerPolicy<T>;

    fn create(&self, color: T::Player) -> Self::P;
}
/**
 * A dynamic policy builder.
 */
pub trait DynMultiplayerPolicyBuilder<'a, T: Game>: Display {
    fn create(&self, color: T::Player) -> Box<dyn MultiplayerPolicy<T> + 'a>;
}

impl<'a, G, PB> DynMultiplayerPolicyBuilder<'a, G> for PB
where
    G: Game,
    PB: MultiplayerPolicyBuilder<G>,
    PB::P: 'a,
{
    fn create(&self, color: G::Player) -> Box<dyn MultiplayerPolicy<G> + 'a> {
        Box::new(PB::create(self, color))
    }
}

/* SINGLEPLAYER POLICY TRAITS */
/* TODO: rename as PlannedPolicy
*/
pub trait SingleplayerPolicyBuilder<T: Game> {
    type P: SingleplayerPolicy<T>;

    fn create(&self) -> Self::P;
}


pub trait SingleplayerPolicy<T: Game> {
    fn solve(&mut self, board: &T) -> Vec<T::Move>;
}

/** MULTIPLAYER POLICIES */

use super::game::NoFeatures;

pub fn get_multi<'a, G: mcts::MCTSGame + 'a>(name: &str) -> Box<dyn DynMultiplayerPolicyBuilder<'a, G> + Sync + 'a> {
    match name {
        "rand" => Box::new(flat::Random::default()),
        "flat" => Box::new(flat::FlatMonteCarlo::default()),
        "flat_ucb" => Box::new(flat::FlatUCBMonteCarlo::default()),
        "uct" => Box::new(mcts::uct::UCT::default()),
        "rave" => Box::new(mcts::rave::RAVE::default()),
        "ppa" => Box::new(ppa::PPA::<_, NoFeatures>::default()),
        "nmcs" => Box::new(nmcs::MultiNMCS::default()),
        _ => panic!("Policy '{}' not found.", name)
    }
}

/* SINGLEPLAYER POLICIES: TODO */
