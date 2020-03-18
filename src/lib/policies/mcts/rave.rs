use crate::game::MultiplayerGame;
use crate::policies::{MultiplayerPolicyBuilder, mcts::{MCTSPolicy, BaseMCTSPolicy, WithMCTSPolicy}};
use crate::settings;

use std::collections::HashMap;
use std::f32;
use std::iter::*;
use std::fmt;
use float_ord::FloatOrd;

/* RAVE */

#[derive(Debug)]
struct MoveInfo {
    wins: f32,
    count: f32,
    wins_AMAF: f32,
    count_AMAF: f32,
}

#[derive(Debug)]
pub struct RAVENodeInfo<G: MultiplayerGame> {
    count: f32,
    moves: HashMap<G::Move, MoveInfo>,
}

pub struct RAVEPolicy_<G: MultiplayerGame> {
    color: G::Player,
    tree: HashMap<usize, RAVENodeInfo<G>>,
    UCT_WEIGHT: f32,
}

impl<G: MultiplayerGame> BaseMCTSPolicy<G> for RAVEPolicy_<G> {
    type NodeInfo = RAVENodeInfo<G>;
    type PlayoutInfo = (bool, Vec<(usize, G::Move)>); // (has_won, history_default)

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        &self.tree
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        &mut self.tree
    }

    fn select_move(&self, board: &G, _exploration: bool) -> G::Move {
        let moves = board.possible_moves();
        // select between optimism and pessimism in the confidence bound.
        let optimistic = board.turn() == self.color;

        let moves_scores = moves.iter().map(|action| {
            let value = self.eval(&board.hash(), &action, optimistic);
            (value, action)
        });

        if board.turn() == self.color {
            *moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        } else {
            *moves_scores.min_by_key(|x| FloatOrd(x.0)).unwrap().1
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        let moves = HashMap::from_iter(board.possible_moves().into_iter().map(|m| {
            (
                *m,
                MoveInfo {
                    wins: 0.,
                    wins_AMAF: 0.,
                    count: 0.,
                    count_AMAF: 0.,
                },
            )
        }));
        RAVENodeInfo { count: 0., moves }
    }

    fn backpropagate(
        &mut self,
        history: Vec<(G, G::Move)>,
        (has_won, history_default): Self::PlayoutInfo,
    ) {
        let z = if has_won { 1. } else { 0. };
        let whole_history = [history.iter().map(|(a,b)| (a.hash(), *b)).collect(), history_default].concat();
        for (t, (state, action)) in history.iter().enumerate() {
            let mut node = self.tree.get_mut(&state.hash()).unwrap();
            node.count += 1.;

            let mut v = node.moves.get_mut(action).unwrap();
            (*v).count += 1.;
            (*v).wins += (z - (*v).wins) / (*v).count;

            // compute AMAF statistics
            for u in (t + 2..whole_history.len()).step_by(2) {
                let (_, action_u) = whole_history[u];
                if (t..u).step_by(2).all(|i| action_u != whole_history[i].1) {
                    if let Some(mut v_amaf) = node.moves.get_mut(&action_u) {
                        (*v_amaf).count_AMAF += 1.;
                        (*v_amaf).wins_AMAF += (z - (*v_amaf).wins_AMAF) / (*v_amaf).count_AMAF;
                    }
                }
            }
        }
    }
}

impl<G: MultiplayerGame> MCTSPolicy<G> for RAVEPolicy_<G> {
    fn simulate(&self, history: &[G]) -> Self::PlayoutInfo {
        let (s, default) = history.last().unwrap().playout_board_history();
        (s.has_won(self.color), default)
    }
}

impl<G: MultiplayerGame> RAVEPolicy_<G> {
    fn beta(v: &MoveInfo) -> f32 {
        let b = 0.0001;
        let mut div = v.count_AMAF + v.count + 4. * v.count_AMAF * v.count * b * b;
        if div == 0. {
            div = 1.
        };
        v.count_AMAF / div
    }

    fn eval(self: &RAVEPolicy_<G>, state: &usize, action: &G::Move, optimistic: bool) -> f32 {
        let node_info = self.tree.get(state).unwrap();
        let v = node_info.moves.get(action).unwrap();

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
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS 
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

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for RAVE {
    type P = RAVEPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(
            RAVEPolicy_ {
                color,
                tree: HashMap::new(),
                UCT_WEIGHT: self.UCT_WEIGHT,
            },
            self.N_PLAYOUTS
        )
    }
}
