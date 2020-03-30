use crate::game::{Game, NoFeatures};

use std::fmt::Display;

/**
 *  Policies that doesn't perform any tree search.
 */
pub mod flat;
/**
 *  Monte-Carlo Tree Search (MCTS) based policies.
 */
pub mod mcts;
/**
 *  Nested Monte-Carlo Search policy.
 */
pub mod nmcs;
/**
 *  Nested Rollout Policy Adaptation.
 */
pub mod nrpa;
/**
 *  Playout Policy Adaptation.
 */ 
pub mod ppa;

/* MULTIPLAYER POLICY TRAITS */

/**
 * A static policy.
 */
pub trait MultiplayerPolicy<T: Game> {
    /**
     *  Chooses the next action given the current game state.
     */
    fn play(&mut self, board: &T) -> T::Move;
}
/**
 * A static policy builder.
 */
pub trait MultiplayerPolicyBuilder<T: Game>: Display {
    /**
     *  Created policy type.
     */
    type P: MultiplayerPolicy<T>;

    /**
     *  Initializes a new policy instance for player `color`.
     */
    fn create(&self, color: T::Player) -> Self::P;
}
/**
 * A dynamic policy builder.
 */
pub trait DynMultiplayerPolicyBuilder<'a, T: Game>: Display {
    /**
     *  Initializes a new policy instance for player `color`, but *dynamically*.
     */
    fn create(&self, color: T::Player) -> Box<dyn MultiplayerPolicy<T> + 'a>;
}

/// Converts a static policy builder to a dynamic one.
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

/// Single-player policy builder.
pub trait SingleplayerPolicyBuilder<T: Game> {
    /// Single player policy type.
    type P: SingleplayerPolicy<T>;

    /// Initializes a new policy instance for the single-player game.
    fn create(&self) -> Self::P;
}

/// Single-player policy.
pub trait SingleplayerPolicy<T: Game> {
    /// Plans the sequence of moves to finish the game.
    fn solve(&mut self, board: &T) -> Vec<T::Move>;
}

/* MULTIPLAYER POLICIES */

use super::game;
/// Dynamically map policy names to policy builder instances.
pub fn get_multi<'a, G>(name: &str) -> Box<dyn DynMultiplayerPolicyBuilder<'a, G> + Sync + 'a> 
where
    G: mcts::MCTSGame + game::SingleWinner + 'a + std::hash::Hash + Eq,
    G::Move: Send
{
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
