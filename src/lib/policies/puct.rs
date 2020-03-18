
use std::f32;
use std::iter::*;

use crate::game;
use super::{MultiplayerPolicyBuilder, N_PLAYOUTS};
use super::mcts::{MCTSPolicy, WithMCTSPolicy};

use std::collections::HashMap;
use std::marker::PhantomData;

use ndarray::Array;

use float_ord::FloatOrd;

/*
 puct::PUCT is deprecated. Use async::puct::PUCT instead.
 */

#[derive(Debug)]
pub struct PUCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
    pub pi: f32,
}

#[derive(Debug)]
pub struct PUCTNodeInfo<G: game::Feature> {
    pub count: f32,
    pub moves: HashMap<G::Move, PUCTMoveInfo>,
}

pub trait Evaluator<G: game::Feature>: Fn(G::Player, &[G]) -> (Array<f32, G::ActionDim>, f32) {}
impl<G:game::Feature, F: Fn(G::Player, &[G]) -> (Array<f32, G::ActionDim>, f32)> Evaluator <G>for F {}

pub struct PUCTPolicy_<'a, G: game::Feature, F: Evaluator<G>> {
    pub color: G::Player,
    pub C_PUCT: f32,
    pub N_HISTORY: usize,
    pub evaluate: &'a F,
    pub tree: HashMap<usize, PUCTNodeInfo<G>>,
}

impl<'a, G: game::Feature, F: Evaluator<G>> MCTSPolicy<G> for PUCTPolicy_<'a, G, F> {
    type NodeInfo = PUCTNodeInfo<G>;
    type PlayoutInfo = (Option<HashMap<G::Move, f32>>, f32, G); // (policy, value, leaf_state).

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        &self.tree
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        &mut self.tree
    }

    fn select_move(&self, board: &G, _exploration: bool) -> G::Move {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board.hash()).unwrap();
        let N = node_info.count;

        let moves_scores = moves.iter().map(|action| {
            let v = node_info.moves.get(action).unwrap();
            let value = v.Q + self.C_PUCT * v.pi * (N.sqrt() / (v.N_a + 1.));
            (value, action)
        });

        if board.turn() == self.color {
            *moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        } else {
            *moves_scores.min_by_key(|x| FloatOrd(x.0)).unwrap().1
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        let n_moves = board.possible_moves().len();
        let moves = HashMap::from_iter(board.possible_moves().into_iter().map(|m| {
            (
                *m,
                PUCTMoveInfo {
                    Q: 0.,
                    N_a: 0.,
                    pi: 1. / (n_moves as f32),
                },
            )
        }));
        PUCTNodeInfo { count: 0., moves }
    }

    fn simulate(&self, history: &[G]) -> Self::PlayoutInfo {
        let board = history.last().unwrap();
        if !board.is_finished() {
            let history: Vec<G> = if history.len() < self.N_HISTORY {
                let mut _h = vec![history.first().unwrap().clone(); self.N_HISTORY - history.len()];
                _h.extend(history.iter().cloned());
                _h
            } else {
                Vec::from(&history[(history.len()-self.N_HISTORY)..(history.len())])
            };
            let (policy, value) = (self.evaluate)(self.color, &history);
            let policy = board.feature_to_moves(&policy);
            (Some(policy), value, board.clone())
        } else {
            if board.has_won(self.color) {
                (None, 1., board.clone())
            } else {
                (None, 0., board.clone())
            }
        }
    }

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, (policy, value, board): Self::PlayoutInfo) {
        if let Some(policy) = policy { // save probabilities of newly created node.
            let z: f32 = board
                .possible_moves()
                .into_iter()
                .map(|m| policy.get(&m).unwrap())
                .sum();
            let z = if z == 0. { 1. } else { z };
            
            for (m, info) in self.tree.get_mut(&board.hash()).unwrap().moves.iter_mut() {
                info.pi = policy.get(&m).unwrap() / z;
            }
            ;
        };

        let value = if board.turn() == self.color { value } else { 1. - value };

        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(&state.hash()).unwrap();
            node.count += 1.;
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (value - (*v).Q) / (*v).N_a;
        }
    }
}


pub type PUCTPolicy<'a,G,F> = WithMCTSPolicy<G, PUCTPolicy_<'a,G,F>> ;


// POLICY BUILDER

pub struct PUCT<'a, G, F> 
    where 
    G: game::Feature,
    F: Fn(G::Player, &[G]) -> (Array<f32, G::ActionDim>, f32)
{
    pub C_PUCT: f32,
    pub N_HISTORY: usize,
    pub evaluate: &'a F,
    pub _g: PhantomData<fn() -> G>,
}

impl<G: game::Feature, F: Evaluator<G>> Copy for PUCT<'_, G, F> {}

impl<G: game::Feature, F: Evaluator<G>> Clone for PUCT<'_, G, F> {
    fn clone(&self) -> Self {
        *self
    }
}

use std::fmt;
impl<G: game::Feature, F: Evaluator<G>> fmt::Display
    for PUCT<'_, G, F>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PUCT")?;
        writeln!(f, "|| C_PUCT: {}", self.C_PUCT)?;
        writeln!(f, "|| N_PLAYOUTS: {}", N_PLAYOUTS)
    }
}

impl<'a, G: game::Feature, F: Evaluator<G>> MultiplayerPolicyBuilder<G>
    for PUCT<'a, G, F>
{
    type P = PUCTPolicy<'a, G, F>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(PUCTPolicy_::<'a, G, F> {
            color,
            C_PUCT: self.C_PUCT,
            N_HISTORY: self.N_HISTORY,
            evaluate: self.evaluate,
            tree: HashMap::new(),
        })
    }
}
