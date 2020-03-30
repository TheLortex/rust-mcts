use crate::game::*;

use ndarray::{Array, Dimension};

/// Output from the dynamics network.
pub struct DynamicsNetworkOutput<H: Dimension> {
    /// Predicted reward.
    pub reward: f32,
    /// Predicted next state.
    pub hidden_state: Array<f32, H>,
}

/// Inference function for the representation network.
pub trait RepresentationEvaluator<G, H>: Fn(Array<f32, G::StateDim>) -> Array<f32, H>
where
    G: Feature,
    H: Dimension,
{
}

impl<G, H, I> RepresentationEvaluator<G, H> for I
where
    G: Feature,
    H: Dimension,
    I: Fn(Array<f32, G::StateDim>) -> Array<f32, H>,
{
}

/// Inference function for the dynamics network.
pub trait DynamicsEvaluator<G, H>:
    Clone + Fn(&Array<f32, H>, &Array<f32, G::ActionDim>) -> DynamicsNetworkOutput<H>
where
    G: Feature,
    H: Dimension,
{
}
impl<G, H, I> DynamicsEvaluator<G, H> for I
where
    G: Feature,
    H: Dimension,
    I: Clone + Fn(&Array<f32, H>, &Array<f32, G::ActionDim>) -> DynamicsNetworkOutput<H>,
{
}

/// Simulated game
pub struct Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H>,
{
    turn: G::Player,
    hidden_state: Array<f32, H>,
    possible_moves: Vec<G::Move>,
    total_reward: f32,
    dynamics_evaluator: DE,
    hidden_dimension: H,
}

impl<G, H, DE> Clone for Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H>,
{
    fn clone(&self) -> Self {
        Self {
            turn: self.turn,
            possible_moves: self.possible_moves.clone(),
            hidden_dimension: self.hidden_dimension.clone(),
            hidden_state: self.hidden_state.clone(),
            dynamics_evaluator: self.dynamics_evaluator.clone(),
            total_reward: self.total_reward,
        }
    }
}

impl<G, H, DE> Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H>,
{
    /// Instanciate a new simulated game.
    ///
    /// # Params
    ///
    /// - `turn`: starting player.
    /// - `hidden_state`: initial hidden state.
    /// - `initial_possible_moves`: available moves for the initial state.
    /// - `dynamics_evaluator`: evaluator for the dynamics network.
    pub fn new(
        turn: G::Player,
        hidden_state: Array<f32, H>,
        initial_possible_moves: Vec<G::Move>,
        dynamics_evaluator: DE,
    ) -> Self {
        let hidden_dimension = hidden_state.raw_dim();
        Simulated {
            turn,
            possible_moves: initial_possible_moves,
            hidden_state,
            total_reward: 0.,
            hidden_dimension,
            dynamics_evaluator,
        }
    }
}

use std::fmt::*;

impl<G, H, DE> Debug for Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Simulated: {:?}\n{:?}",
            self.total_reward, self.hidden_state
        )
    }
}

impl<G, H, SD> Base for Simulated<G, H, SD>
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>,
{
    type Move = G::Move;

    fn possible_moves(&self) -> Vec<Self::Move> {
        self.possible_moves.clone()
    }
}

impl<G, H, SD> Playable for Simulated<G, H, SD>
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>,
{
    fn play(&mut self, action: &Self::Move) -> f32 {
        let mut move_as_prob: HashMap<Self::Move, f32> = HashMap::new();
        move_as_prob.insert(*action, 1.);
        let move_encoded = G::moves_to_feature(&move_as_prob);

        let network_output = (self.dynamics_evaluator)(&self.hidden_state, &move_encoded);
        self.hidden_state = network_output.hidden_state;

        self.possible_moves = G::all_possible_moves().to_vec();

        // set next player
        self.turn = G::player_after(self.turn);

        network_output.reward
    }
}

impl<G, H, SD> Game for Simulated<G, H, SD>
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>,
{
    type Player = G::Player;

    fn players() -> Vec<Self::Player> {
        G::players()
    }

    fn player_after(player: Self::Player) -> Self::Player {
        G::player_after(player)
    }

    fn turn(&self) -> Self::Player {
        self.turn
    }
}

impl<G, H, SD> Feature for Simulated<G, H, SD>
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>,
{
    type StateDim = H;
    type ActionDim = G::ActionDim;

    fn state_dimension(&self) -> Self::StateDim {
        self.hidden_dimension.clone()
    }

    fn state_to_feature(&self, _pov: Self::Player) -> Array<f32, Self::StateDim> {
        self.hidden_state.clone()
    }

    fn action_dimension() -> Self::ActionDim {
        G::action_dimension()
    }

    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim> {
        G::moves_to_feature(moves)
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        G::all_feature_to_moves(features)
    }

    fn all_possible_moves() -> Vec<Self::Move> {
        G::all_possible_moves()
    }

    fn all_feature_to_moves(features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        G::all_feature_to_moves(features)
    }
}
