use rand::seq::SliceRandom;
use std::f32;
use std::iter::*;

use super::super::game::{Game, MoveCode};
use super::mcts::{UCTMoveInfo, UCTNodeInfo};
use super::{Policy, PolicyBuilder, N_PLAYOUTS};

use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct PUCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
    pub pi: f32,
}

#[derive(Debug)]
pub struct PUCTNodeInfo<G: Game> {
    pub count: f32,
    pub moves: HashMap<G::Move, PUCTMoveInfo>,
}

pub struct PUCTPolicy<'a, G: Game, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> {
    pub color: G::Player,
    pub C_PUCT: f32,
    pub evaluate: &'a F,
    pub tree: HashMap<G::GameHash, PUCTNodeInfo<G>>,
}

impl<'a, G: Game, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> PUCTPolicy<'a, G, F> {
    fn simulate(self: &mut PUCTPolicy<'a, G, F>, board: &G) {
        let mut b = board.clone();
        let history = self.select(&mut b);
        let value = if b.winner().is_none() {
            let (policy, value) = (self.evaluate)(&b);
            self.expand(&b, policy);
            value
        } else {
            b.score(self.color)
        };
        self.backup(history, value);
    }

    fn select(self: &mut PUCTPolicy<'a, G, F>, b: &mut G) -> Vec<(G::GameHash, G::Move)> {
        let mut history: Vec<(G::GameHash, G::Move)> = Vec::new();

        while { b.winner() == None } {
            let s_t = b.hash();
            match self.tree.get(&s_t) {
                None => {
                    return history;
                }
                Some(_node) => {
                    let a = self.select_move(&b);
                    history.push((s_t, a));
                    b.play(&a)
                }
            };
        }
        history
    }

    // Aggressive selection algorithm.
    fn select_move(self: &PUCTPolicy<'a, G, F>, board: &G) -> G::Move {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board.hash()).unwrap();

        let N = node_info.count;
        let mut max_move = None;
        let mut max_value = 0.;
        for _move in moves.iter() {
            let v = node_info.moves.get(_move).unwrap();
            let value = v.Q + self.C_PUCT * v.pi * (N.sqrt() / (v.N_a + 1.));
            if value >= max_value {
                max_value = value;
                max_move = Some(*_move);
            }
        }
        max_move.unwrap()
    }

    fn expand(self: &mut PUCTPolicy<'a, G, F>, board: &G, policy: HashMap<G::Move, f32>) {
        // normalize policy probabilities among possible moves.
        let z: f32 = board.possible_moves().into_iter().map(|m| {policy.get(&m).unwrap()}).sum();
        let z = if z == 0. { 1. } else { z };
        let moves = HashMap::from_iter(board.possible_moves().into_iter().map(|m| {
            (
                *m,
                PUCTMoveInfo {
                    Q: 0.,
                    N_a: 0.,
                    pi: policy.get(&m).unwrap() / z,
                },
            )
        }));

        self.tree
            .insert(board.hash(), PUCTNodeInfo { count: 0., moves });
    }

    fn backup(
        self: &mut PUCTPolicy<'a, G, F>,
        history: Vec<(G::GameHash, G::Move)>,
        value: f32,
    ) {
        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;
            
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q   += (value - (*v).Q) / (*v).N_a;
        }
    }
}

impl<'a, G: Game, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> Policy<G> for PUCTPolicy<'a, G, F> {
    fn play(self: &mut PUCTPolicy<'a, G, F>, board: &G) -> G::Move {

        for _ in 0..N_PLAYOUTS {
            self.simulate(board)
        }

        let mut best_move = None;
        let mut max_visited = 0.;

        let node_info = self.tree.get(&board.hash()).unwrap();

        for m in board.possible_moves().iter() {
            let v = node_info.moves.get(&m).unwrap();
            let value = v.Q;

            if value >= max_visited {
                max_visited = value;
                best_move = Some(*m);
            }
        }
        best_move.unwrap()
    }
}
