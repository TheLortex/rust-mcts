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

/* RAVE */

#[derive(Debug,Clone,Copy)]
pub struct RAVEMoveInfo {
    wins: f32,
    count: f32,
    wins_AMAF: f32,
    count_AMAF: f32,
}

#[derive(Debug,Clone,Copy)]
pub struct RAVENodeInfo {
    count: f32,
}

pub struct RAVEPolicy_<G: Game> {
    color: G::Player,
    UCT_WEIGHT: f32,
}

impl<G: super::MCTSGame> BaseMCTSPolicy<G> for RAVEPolicy_<G> {
    type NodeInfo = RAVENodeInfo;
    type MoveInfo = RAVEMoveInfo;
    type PlayoutInfo = (bool, Vec<G::Move>); // (has_won, history_default)

    fn get_value(
        &self,
        board: &G,
        action: &G::Move,
        node_info: &Self::NodeInfo,
        move_info: &Self::MoveInfo,
        exploration: bool,
    ) -> f32 {
        let optimistic = board.turn() == self.color;
        let value = self.eval(node_info, move_info, optimistic);
        let multiplier = if board.turn() == self.color { 1. } else { -1. };
        multiplier * value
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        RAVENodeInfo { count: 0. }
    }

    fn default_move(&self, board: &G, action: &G::Move) -> Self::MoveInfo {
        RAVEMoveInfo {
            wins: 0.,
            wins_AMAF: 0.,
            count: 0.,
            count_AMAF: 0.,
        }
    }

    fn backpropagate_new_node(
        &self,
        node: &mut MCTSNode<G, Self>,
        history: &[G::Move],
        playout: &Self::PlayoutInfo,
    ) {
    }

    fn backpropagate(
        &self,
        index: usize,
        info: &mut MCTSNode<G, Self>,
        action: &G::Move,
        history: &[G::Move],
        (has_won, history_default): &Self::PlayoutInfo,
    ) {
        let z = if *has_won { 1. } else { 0. };

        info.node.count += 1.;

        let move_info = info.moves.get_mut(action).unwrap();
        move_info.count += 1.;
        move_info.wins += (z - move_info.wins) / move_info.count;

        let whole_history = [history, &history_default].concat();

        for u in (index + 2..whole_history.len()).step_by(2) {
            let action_u = whole_history[u];
            if (index..u).step_by(2).all(|i| action_u != whole_history[i]) {
                if let Some(mut v_amaf) = info.moves.get_mut(&action_u) {
                    (*v_amaf).count_AMAF += 1.;
                    (*v_amaf).wins_AMAF += (z - (*v_amaf).wins_AMAF) / (*v_amaf).count_AMAF;
                }
            }
        }
    }

    fn simulate(&self, board: &G) -> <Self as BaseMCTSPolicy<G>>::PlayoutInfo {
        let (s, default) = board.playout_history();
        let default: Vec<G::Move> = default.iter().map(|(_,m)| *m).collect();
        (s.has_won(self.color), default)
    }
}

impl<G: super::MCTSGame> RAVEPolicy_<G> {
    fn beta(v: &RAVEMoveInfo) -> f32 {
        let b = 0.0001;
        let mut div = v.count_AMAF + v.count + 4. * v.count_AMAF * v.count * b * b;
        if div == 0. {
            div = 1.
        };
        v.count_AMAF / div
    }

    fn eval(
        self: &RAVEPolicy_<G>,
        node_info: &RAVENodeInfo,
        v: &RAVEMoveInfo,
        optimistic: bool,
    ) -> f32 {
        let multiplier = if optimistic { 1. } else { -1. };
        let v_mean =
            v.wins + multiplier * self.UCT_WEIGHT * (node_info.count.ln() / (1. + v.count)).sqrt();
        let v_AMAF = v.wins_AMAF;

        let beta = Self::beta(v);
        (1. - beta) * v_mean + beta * v_AMAF
    }
}

pub type RAVEPolicy<G> = WithMCTSPolicy<G, RAVEPolicy_<G>>;

pub struct RAVE {
    UCT_WEIGHT: f32,
    N_PLAYOUTS: usize,
}

impl Default for RAVE {
    fn default() -> RAVE {
        RAVE {
            UCT_WEIGHT: 0.4,
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        }
    }
}

impl fmt::Display for RAVE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RAVE")?;
        writeln!(f, "|| UCT_WEIGHT: {}", self.UCT_WEIGHT)?;
        writeln!(f, "|| N_PLAYOUT: {}", self.N_PLAYOUTS)
    }
}

impl<G> MultiplayerPolicyBuilder<G> for RAVE
where
    G::Move: Send,
    G::Player: Send,
    G: super::MCTSGame
{
    type P = RAVEPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(
            RAVEPolicy_ {
                color,
                UCT_WEIGHT: self.UCT_WEIGHT,
            },
            self.N_PLAYOUTS
        )
    }
}

