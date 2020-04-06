use crate::game::{Game, NoFeatures};
use crate::settings;

use async_trait::async_trait;
use std::fmt::Display;

///
/// Policies that doesn't perform any tree search.
///
pub mod flat;
///
/// Monte-Carlo Tree Search (MCTS) based policies.
///
pub mod mcts;
///
/// Nested Monte-Carlo Search policy.
///
pub mod nmcs;
///
/// Nested Rollout Policy Adaptation.
///
pub mod nrpa;
///
/// Playout Policy Adaptation.
///
pub mod ppa;

/* MULTIPLAYER POLICY TRAITS */

///
///A static policy.
///
#[async_trait]
pub trait MultiplayerPolicy<T: Game> {
    ///
    /// Chooses the next action given the current game state.
    ///
    async fn play(&mut self, board: &T) -> T::Move;
}
///
///A static policy builder.
///
pub trait MultiplayerPolicyBuilder<T: Game>: Display {
    ///
    /// Created policy type.
    ///
    type P: MultiplayerPolicy<T>;

    ///
    /// Initializes a new policy instance for player `color`.
    ///
    fn create(&self, color: T::Player) -> Self::P;
}
///
///A dynamic policy builder.
///
pub trait DynMultiplayerPolicyBuilder<'a, T: Game>: Display {
    ///
    /// Initializes a new policy instance for player `color`, but *dynamically*.
    ///
    fn create(&self, color: T::Player) -> Box<dyn MultiplayerPolicy<T> + Send + Sync + 'a>;
}

/// Converts a static policy builder to a dynamic one.
impl<'a, G, PB> DynMultiplayerPolicyBuilder<'a, G> for PB
where
    G: Game,
    PB: MultiplayerPolicyBuilder<G>,
    PB::P: 'a + Send + Sync,
{
    fn create(&self, color: G::Player) -> Box<dyn MultiplayerPolicy<G> + Send + Sync + 'a> {
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
#[async_trait]
pub trait SingleplayerPolicy<T: Game> {
    /// Plans the sequence of moves to finish the game.
    async fn solve(&mut self, board: &T) -> Vec<T::Move>;
}

/* MULTIPLAYER POLICIES */

use super::game;
/// Dynamically map policy names to policy builder instances.
pub fn get_multi<'a, G>(
    config: settings::Config,
    name: &str,
) -> Box<dyn DynMultiplayerPolicyBuilder<'a, G> + Sync + Send + 'a>
where
    G: mcts::MCTSGame + game::SingleWinner + 'a + std::hash::Hash + Eq,
    G::Move: Send,
{
    match name {
        "rand" => Box::new(flat::Random {}),
        "flat" => Box::new(config.policies.flat),
        "flat_ucb" => Box::new(config.policies.flat_ucb),
        "uct" => Box::new(config.policies.uct),
        "rave" => Box::new(config.policies.rave),
        "ppa" => Box::new(ppa::PPA::<_, NoFeatures>::new(config.policies.ppa)),
        "nmcs" => Box::new(nmcs::MultiNMCS::default()),
        _ => panic!("Policy '{}' not found.", name),
    }
}
