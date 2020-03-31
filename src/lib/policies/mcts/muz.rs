use super::puct::{Evaluator, PUCTPolicy, PUCTSettings, PUCT};
use crate::game;
use crate::game::meta::simulated::{
    DynamicsEvaluator, RepresentationEvaluator, Simulated,
};
use crate::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};

use ndarray::Dimension;
use std::marker::PhantomData;
use std::sync::Arc;

/// MuZero policy
pub struct MuzPolicy<G, H>
where
    G: game::Feature + 'static,
    H: Dimension,
{
    player: G::Player,
    /// PUCT policy instance. Can be taken to gather statistics.
    pub mcts: Option<PUCTPolicy<Simulated<G, H>>>,
    config: Muz<G, H>,
}

impl<G, H> MultiplayerPolicy<G> for MuzPolicy<G, H>
where
    G: game::Feature + 'static,
    H: Dimension,
{
    fn play(&mut self, board: &G) -> G::Move {
        let net_output = (self.config.representation_evaluate)(board.state_to_feature(self.player));
        let simulator = Simulated::new(
            board.turn(),
            net_output,
            board.possible_moves(),
            self.config.dynamics_evaluate.clone(),
        );

        let mcts_policy_builder: PUCT<Simulated<G, H>> = PUCT {
            _g: PhantomData,
            evaluate: self.config.prediction_evaluate.clone(),
            config: self.config.puct,
            N_PLAYOUTS: self.config.N_PLAYOUTS,
        };

        let mut mcts_policy = mcts_policy_builder.create(self.player);

        let action = mcts_policy.play(&simulator);
        self.mcts = Some(mcts_policy);
        action
    }
}

/// MuZero policy builder.
pub struct Muz<G, H>
where
    G: game::Feature + 'static,
    H: Dimension,
{
    /// PUCT settings.
    pub puct: PUCTSettings,
    /// Number of PUCT playouts per move.
    pub N_PLAYOUTS: usize,
    /// Evaluator for the prediction network.
    pub prediction_evaluate: Arc<dyn Evaluator<Simulated<G, H>>>,
    /// Evaluator for the representation network.
    pub representation_evaluate: Arc<dyn RepresentationEvaluator<G, H>>,
    /// Evaluator for the dynamics network.
    pub dynamics_evaluate: Arc<dyn DynamicsEvaluator<G, H>>,
    /// PhantomData storing game type information.
    pub _g: PhantomData<fn() -> G>,
    /// PhantomData storing dimension type information.
    pub _h: PhantomData<fn() -> H>,
}

use std::fmt;

impl<G, H> fmt::Display for Muz<G, H>
where
    G: game::Feature + 'static,
    H: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MUZ")?;
        writeln!(f, "||{:?}", self.puct)
    }
}

impl<G, H> Clone for Muz<G, H>
where
    G: game::Feature + 'static,
    H: Dimension,
{
    fn clone(&self) -> Self {
        Muz {
            puct: self.puct,
            N_PLAYOUTS: self.N_PLAYOUTS,
            prediction_evaluate: self.prediction_evaluate.clone(),
            representation_evaluate: self.representation_evaluate.clone(),
            dynamics_evaluate: self.dynamics_evaluate.clone(),
            _h: PhantomData,
            _g: PhantomData,
        }
    }
}

impl<G, H> MultiplayerPolicyBuilder<G> for Muz<G, H>
where
    G: game::Feature + 'static,
    H: Dimension,
{
    type P = MuzPolicy<G, H>;

    fn create(&self, color: G::Player) -> Self::P {
        MuzPolicy {
            player: color,
            config: self.clone(),
            mcts: None,
        }
    }
}
