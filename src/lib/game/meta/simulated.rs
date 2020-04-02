use crate::deep::evaluator::{dynamics, DynamicsEvaluatorChannel};
use crate::game::*;

use ndarray::{Array, Dimension};
use tokio::sync::mpsc;
use ndarray::Ix3;


/// Output from the dynamics network.
pub struct DynamicsNetworkOutput<H: Dimension> {
    /// Predicted reward.
    pub reward: f32,
    /// Predicted next state.
    pub repr_state: Array<f32, H>,
}

/// Simulated game
pub struct Simulated<G>
where
    G: Feature,
{
    turn: G::Player,
    repr_state: Array<f32, Ix3>,
    possible_moves: Vec<G::Move>,
    total_reward: f32,
    dynamics_evaluator: mpsc::Sender<DynamicsEvaluatorChannel>,
    repr_dimension: Ix3,
}

impl<G> Clone for Simulated<G>
where
    G: Feature,
{
    fn clone(&self) -> Self {
        Self {
            turn: self.turn,
            possible_moves: self.possible_moves.clone(),
            repr_state: self.repr_state.clone(),
            dynamics_evaluator: self.dynamics_evaluator.clone(),
            total_reward: self.total_reward,
            repr_dimension: self.repr_dimension,
        }
    }
}

impl<G> Simulated<G>
where
    G: Feature,
{
    /// Instanciate a new simulated game.
    ///
    /// # Params
    ///
    /// - `turn`: starting player.
    /// - `repr_state`: initial repr state.
    /// - `initial_possible_moves`: available moves for the initial state.
    /// - `dynamics_evaluator`: evaluator for the dynamics network.
    pub fn new(
        turn: G::Player,
        repr_state: Array<f32, Ix3>,
        initial_possible_moves: Vec<G::Move>,
        dynamics_evaluator: mpsc::Sender<DynamicsEvaluatorChannel>,
    ) -> Self {
        let repr_dimension = repr_state.raw_dim();
        Simulated {
            turn,
            possible_moves: initial_possible_moves,
            repr_state,
            total_reward: 0.,
            dynamics_evaluator,
            repr_dimension,
        }
    }
}

use std::fmt::*;

impl<G> Debug for Simulated<G>
where
    G: Feature,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Simulated: {:?}\n{:?}",
            self.total_reward, self.repr_state
        )
    }
}

impl<G> Base for Simulated<G>
where
    G: Feature + 'static,
{
    type Move = G::Move;

    fn possible_moves(&self) -> Vec<Self::Move> {
        self.possible_moves.clone()
    }
}

#[async_trait]
impl<G> Playable for Simulated<G>
where
    G: Feature + 'static,
{
    async fn play(&mut self, action: &<Self as Base>::Move) -> f32 {
        let mut move_as_prob: HashMap<<Self as Base>::Move, f32> = HashMap::new();
        move_as_prob.insert(*action, 1.);
        let move_encoded = G::moves_to_feature(&move_as_prob);

        let network_output = dynamics(
            self.dynamics_evaluator.clone(),
            &self.repr_state,
            &move_encoded,
            true,
        )
        .await;
        self.repr_state = network_output.repr_state;

        self.possible_moves = G::all_possible_moves().to_vec();

        // set next player
        self.turn = G::player_after(self.turn);

        network_output.reward
    }
}

impl<G> Game for Simulated<G>
where
    G: Feature + 'static,
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

impl<G> Feature for Simulated<G>
where
    G: Feature + 'static,
{
    type StateDim = Ix3;
    type ActionDim = G::ActionDim;

    fn state_dimension(&self) -> Self::StateDim {
        self.repr_dimension
    }

    fn state_to_feature(&self, _pov: Self::Player) -> Array<f32, Self::StateDim> {
        self.repr_state.clone()
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
