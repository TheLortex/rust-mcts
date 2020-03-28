use super::puct::{Evaluator, PUCTSettings, PUCT, PUCTPolicy};
use crate::game;
use crate::game::meta::simulated::{DynamicsEvaluator, Simulated, NetworkOutput, RepresentationEvaluator};
use crate::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};
use ndarray::Array;

use ndarray::Dimension;
use std::marker::PhantomData;

pub struct MuzPolicy<G, H, HE, SE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: RepresentationEvaluator<G, H>,
    DE: DynamicsEvaluator<G, H>,
    HE: Evaluator<Simulated<G, H, DE>>,
{
    player: G::Player,
    pub mcts: Option<PUCTPolicy<Simulated<G, H, DE>, HE>>,

    config: Muz<G,H,HE,SE,DE>,
}

impl<G, H, HE, SE, DE> MultiplayerPolicy<G> for MuzPolicy<G, H, HE, SE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: RepresentationEvaluator<G, H>,
    DE: DynamicsEvaluator<G, H>,
    HE: Evaluator<Simulated<G, H, DE>> + Clone,
{
    fn play(&mut self, board: &G) -> G::Move {
        let net_output = (self.config.state_evaluate)(board.state_to_feature(self.player));
        let simulator = Simulated::new(
            board.turn(),
            net_output,
            board.possible_moves(),
            self.config.dynamics_evaluate.clone(),
        );

        let mcts_policy_builder: PUCT<Simulated<G, H, DE>, HE> = PUCT {
            _g: PhantomData,
            evaluate: self.config.hidden_evaluate.clone(),
            config: self.config.puct,
            N_PLAYOUTS: self.config.N_PLAYOUTS,
        };

        let mut mcts_policy = mcts_policy_builder.create(self.player);

        let action = mcts_policy.play(&simulator);
        self.mcts = Some(mcts_policy);
        action
    }
}

pub struct Muz<G, H, HE, SE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: Fn(Array<f32, G::StateDim>) -> Array<f32, H>,
    DE: Fn(&Array<f32, H>, &Array<f32, G::ActionDim>) -> NetworkOutput<H> + Clone,
    HE: Fn(G::Player, &Simulated<G, H, DE>) -> (Array<f32, G::ActionDim>, f32),
{
    pub puct: PUCTSettings,
    pub N_PLAYOUTS: usize,
    pub hidden_evaluate: HE,
    pub state_evaluate: SE,
    pub dynamics_evaluate: DE,
    pub _g: PhantomData<fn() -> G>,
    pub _h: PhantomData<fn() -> H>,
}

use std::fmt;

impl<G, H, HE, SE, DE> fmt::Display for Muz<G, H, HE, SE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: RepresentationEvaluator<G, H>,
    DE: DynamicsEvaluator<G, H>,
    HE: Evaluator<Simulated<G, H, DE>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MUZ")?;
        writeln!(f, "||{:?}", self.puct)
    }
}

impl<G, H, HE, SE, DE> Clone for Muz<G, H, HE, SE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: RepresentationEvaluator<G, H> + Clone,
    DE: DynamicsEvaluator<G, H>,
    HE: Evaluator<Simulated<G, H, DE>> + Clone,
{
    fn clone(&self) -> Self {
        Muz {
            puct: self.puct,
            N_PLAYOUTS: self.N_PLAYOUTS,
            hidden_evaluate: self.hidden_evaluate.clone(),
            state_evaluate: self.state_evaluate.clone(),
            dynamics_evaluate: self.dynamics_evaluate.clone(),
            _h: PhantomData,
            _g: PhantomData,
        }
    }
}



impl<G, H, HE, SE, DE> MultiplayerPolicyBuilder<G> for Muz<G, H, HE, SE, DE>
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: RepresentationEvaluator<G, H> + Clone,
    DE: DynamicsEvaluator<G, H>,
    HE: Evaluator<Simulated<G, H, DE>> + Clone,
{
    type P = MuzPolicy<G, H, HE, SE, DE>;

    fn create(&self, color: G::Player) -> Self::P {
        MuzPolicy {
            player: color,
            config: self.clone(),
            mcts: None,
        }
    }

}