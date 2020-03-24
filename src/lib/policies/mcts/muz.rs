use super::puct::{Evaluator, PUCTSettings, PUCT};
use crate::game::meta::simulated::{StateEvaluator, DynamicsEvaluator, NetworkOutput, Simulated};
use crate::game;
use crate::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};

use std::marker::PhantomData;
use ndarray::Dimension;

struct MuzSettings {
    puct: PUCTSettings,
}

struct MuzPolicy<G,H,HE,SE,DE> 
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: StateEvaluator<G,H>,
    DE: DynamicsEvaluator<G,H>,
    HE: Evaluator<Simulated<G,H,DE>>,
{
    s: MuzSettings,
    player: G::Player,
    hidden_evaluate: HE,
    state_evaluate: SE,
    dynamics_evaluate: DE,
    _g: PhantomData<G>,
    _h: PhantomData<H>
}


impl<G,H,HE,SE,DE> MultiplayerPolicy<G> for MuzPolicy<G,H,HE,SE,DE> 
where
    G: game::Feature + 'static,
    H: Dimension,
    SE: StateEvaluator<G,H>,
    DE: DynamicsEvaluator<G,H>,
    HE: Evaluator<Simulated<G,H,DE>> + Clone,
{ 
    fn play(&mut self, board: &G) -> G::Move {
        let net_output = (self.state_evaluate)(board.state_to_feature(self.player));
        let simulator = Simulated::new(net_output.hidden_state, self.dynamics_evaluate);

        let mcts_policy_builder = PUCT {
            _g: PhantomData,
            evaluate: self.hidden_evaluate,
            s: self.s.puct,
            N_PLAYOUTS: 800 // TODO
        };

        let mcts_policy = mcts_policy_builder.create(self.player);


    }
}