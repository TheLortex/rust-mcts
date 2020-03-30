use super::puct::{Evaluator, PUCTPolicy, PUCTSettings, PUCT};
use crate::game;
use crate::game::meta::simulated::{
    DynamicsEvaluator, DynamicsNetworkOutput, RepresentationEvaluator, Simulated,
};
use crate::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};
use ndarray::Array;

use ndarray::Dimension;
use std::marker::PhantomData;

/// MuZero policy
pub struct MuzPolicy<G, H, PE, RE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    RE: RepresentationEvaluator<G, H>,
    DE: DynamicsEvaluator<G, H>,
    PE: Evaluator<Simulated<G, H, DE>>,
{
    player: G::Player,
    /// PUCT policy instance. Can be taken to gather statistics.
    pub mcts: Option<PUCTPolicy<Simulated<G, H, DE>, PE>>,

    config: Muz<G, H, PE, RE, DE>,
}

impl<G, H, PE, RE, DE> MultiplayerPolicy<G> for MuzPolicy<G, H, PE, RE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    RE: RepresentationEvaluator<G, H>,
    DE: DynamicsEvaluator<G, H>,
    PE: Evaluator<Simulated<G, H, DE>> + Clone,
{
    fn play(&mut self, board: &G) -> G::Move {
        let net_output = (self.config.representation_evaluate)(board.state_to_feature(self.player));
        let simulator = Simulated::new(
            board.turn(),
            net_output,
            board.possible_moves(),
            self.config.dynamics_evaluate.clone(),
        );

        let mcts_policy_builder: PUCT<Simulated<G, H, DE>, PE> = PUCT {
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
pub struct Muz<G, H, PE, RE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    RE: Fn(Array<f32, G::StateDim>) -> Array<f32, H>,
    DE: Fn(&Array<f32, H>, &Array<f32, G::ActionDim>) -> DynamicsNetworkOutput<H> + Clone,
    PE: Fn(G::Player, &Simulated<G, H, DE>) -> (Array<f32, G::ActionDim>, f32),
{
    /// PUCT settings.
    pub puct: PUCTSettings,
    /// Number of PUCT playouts per move.
    pub N_PLAYOUTS: usize,
    /// Evaluator for the prediction network.
    pub prediction_evaluate: PE,
    /// Evaluator for the representation network.
    pub representation_evaluate: RE,
    /// Evaluator for the dynamics network.
    pub dynamics_evaluate: DE,
    /// PhantomData storing game type information.
    pub _g: PhantomData<fn() -> G>,
    /// PhantomData storing dimension type information.
    pub _h: PhantomData<fn() -> H>,
}

use std::fmt;

impl<G, H, PE, RE, DE> fmt::Display for Muz<G, H, PE, RE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    RE: RepresentationEvaluator<G, H>,
    DE: DynamicsEvaluator<G, H>,
    PE: Evaluator<Simulated<G, H, DE>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MUZ")?;
        writeln!(f, "||{:?}", self.puct)
    }
}

impl<G, H, PE, RE, DE> Clone for Muz<G, H, PE, RE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    RE: RepresentationEvaluator<G, H> + Clone,
    DE: DynamicsEvaluator<G, H>,
    PE: Evaluator<Simulated<G, H, DE>> + Clone,
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

impl<G, H, PE, RE, DE> MultiplayerPolicyBuilder<G> for Muz<G, H, PE, RE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    RE: RepresentationEvaluator<G, H> + Clone,
    DE: DynamicsEvaluator<G, H>,
    PE: Evaluator<Simulated<G, H, DE>> + Clone,
{
    type P = MuzPolicy<G, H, PE, RE, DE>;

    fn create(&self, color: G::Player) -> Self::P {
        MuzPolicy {
            player: color,
            config: self.clone(),
            mcts: None,
        }
    }
}
