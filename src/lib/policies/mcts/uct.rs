use crate::game::MultiplayerGame;
use crate::policies::{
    mcts::{BaseMCTSPolicy, MCTSPolicy, WithMCTSPolicy},
    MultiplayerPolicyBuilder,
};
use crate::settings;

use float_ord::FloatOrd;
use std::collections::HashMap;
use std::f32;
use std::fmt;
use std::iter::*;

/* UCT */

#[derive(Debug)]
pub struct UCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
}

#[derive(Debug)]
pub struct UCTNodeInfo<G: MultiplayerGame> {
    pub count: f32,
    pub moves: HashMap<G::Move, UCTMoveInfo>,
}

pub struct UCTPolicy_<G: MultiplayerGame> {
    color: G::Player,
    tree: HashMap<usize, UCTNodeInfo<G>>,
    UCT_WEIGHT: f32,
}

impl<G: MultiplayerGame> BaseMCTSPolicy<G> for UCTPolicy_<G> {
    type NodeInfo = UCTNodeInfo<G>;
    type PlayoutInfo = bool;

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        &self.tree
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        &mut self.tree
    }

    fn select_move(&self, board: &G, exploration: bool) -> G::Move {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board.hash()).unwrap();
        let N = node_info.count;

        // select between optimism and pessimism in the confidence bound.
        let move_cb_multiplier = if board.turn() == self.color { 1. } else { -1. };

        let moves_scores = moves.iter().map(|action| {
            let v = node_info.moves.get(action).unwrap();
            let cb = self.UCT_WEIGHT * (N.ln() / (v.N_a + 1.)).sqrt();
            let value = if exploration {
                v.Q + move_cb_multiplier * cb
            } else {
                v.N_a
            };
            (value, action)
        });

        if board.turn() == self.color {
            *moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        } else {
            *moves_scores.min_by_key(|x| FloatOrd(x.0)).unwrap().1
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        let moves = HashMap::from_iter(
            board
                .possible_moves()
                .into_iter()
                .map(|m| (*m, UCTMoveInfo { Q: 0., N_a: 0. })),
        );

        UCTNodeInfo { count: 0., moves }
    }

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, playout: Self::PlayoutInfo) {
        let z = if playout { 1. } else { 0. };
        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(&state.hash()).unwrap();
            node.count += 1.;

            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (z - (*v).Q) / (*v).N_a;
        }
    }
}

impl<G: MultiplayerGame> MCTSPolicy<G> for UCTPolicy_<G> {
    fn simulate(&self, board: &G) -> Self::PlayoutInfo {
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

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for UCT {
    type P = UCTPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(
            UCTPolicy_ {
                color,
                tree: HashMap::new(),
                UCT_WEIGHT: self.UCT_WEIGHT,
            },
            self.N_PLAYOUTS,
        )
    }
}
