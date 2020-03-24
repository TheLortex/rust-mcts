use crate::game::{Game, Playout};
use crate::policies::{
    mcts::{BaseMCTSPolicy, MCTSNode, WithMCTSPolicy},
    MultiplayerPolicyBuilder,
};
use crate::settings;

use float_ord::FloatOrd;
use std::collections::HashMap;
use std::f32;
use std::fmt;
use std::iter::*;

/* UCT */

#[derive(Debug,Clone,Copy)]
pub struct UCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
}

#[derive(Debug,Clone,Copy)]
pub struct UCTNodeInfo {
    pub count: f32,
}

pub struct UCTPolicy_<G: Game> {
    color: G::Player,
    UCT_WEIGHT: f32,
}

impl<G> BaseMCTSPolicy<G> for UCTPolicy_<G>
where
    G::Move: Send,
    G: super::MCTSGame,
{
    type NodeInfo = UCTNodeInfo;
    type MoveInfo = UCTMoveInfo;
    type PlayoutInfo = bool;

    fn get_value(
        &self,
        board: &G,
        action: &G::Move,
        node_info: &Self::NodeInfo,
        move_info: &Self::MoveInfo,
        exploration: bool,
    ) -> f32 {
        let move_cb_multiplier = if board.turn() == self.color { 1. } else { -1. };

        let N = node_info.count;
        let cb = self.UCT_WEIGHT * (N.ln() / (move_info.N_a + 1.)).sqrt();
        let value = if exploration {
            move_info.Q + move_cb_multiplier * cb
        } else {
            move_info.N_a
        };

        move_cb_multiplier * value
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        UCTNodeInfo { count: 0. }
    }

    fn default_move(&self, board: &G, action: &G::Move) -> Self::MoveInfo {
        UCTMoveInfo { Q: 0., N_a: 0. }
    }

    fn backpropagate(
        &self,
        _index: usize,
        info: &mut MCTSNode<G, Self>,
        action: &G::Move,
        history: &[G::Move],
        playout: &Self::PlayoutInfo,
    ) {
        let z = if *playout { 1. } else { 0. };
        info.node.count += 1.;
        let move_info = info.moves.get_mut(action).unwrap();
        move_info.N_a += 1.;
        move_info.Q += (z - move_info.Q) / move_info.N_a;
    }

    fn backpropagate_new_node(
        &self,
        node: &mut MCTSNode<G, Self>,
        history: &[G::Move],
        playout: &Self::PlayoutInfo,
    ) {
    }

    fn simulate(&self, board: &G) -> <Self as BaseMCTSPolicy<G>>::PlayoutInfo {
        board.playout_board().has_won(self.color)
    }
}

pub type UCTPolicy<G> = WithMCTSPolicy<G, UCTPolicy_<G>>;

pub struct UCT {
    UCT_WEIGHT: f32,
    N_PLAYOUTS: usize,
}

impl Default for UCT {
    fn default() -> Self {
        Self {
            UCT_WEIGHT: 0.4,
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        }
    }
}

impl fmt::Display for UCT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "UCT")?;
        writeln!(f, "|| UCT_WEIGHT: {}", self.UCT_WEIGHT)?;
        writeln!(f, "|| N_PLAYOUT: {}", self.N_PLAYOUTS)
    }
}

impl<G> MultiplayerPolicyBuilder<G> for UCT
where
    G::Move: Send,
    G::Player: Send,
    G: super::MCTSGame,
{
    type P = UCTPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(
            UCTPolicy_ {
                color,
                UCT_WEIGHT: self.UCT_WEIGHT,
            },
            self.N_PLAYOUTS,
        )
    }
}

