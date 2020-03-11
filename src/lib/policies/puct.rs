use std::f32;
use std::iter::*;

use super::super::game::MultiplayerGame;
use super::{MultiplayerPolicy, MultiplayerPolicyBuilder, N_PLAYOUTS};

use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct PUCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
    pub pi: f32,
}

#[derive(Debug)]
pub struct PUCTNodeInfo<G: MultiplayerGame> {
    pub count: f32,
    pub moves: HashMap<G::Move, PUCTMoveInfo>,
}

pub struct PUCTPolicy<'a, G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> {
    pub color: G::Player,
    pub C_PUCT: f32,
    pub evaluate: &'a F,
    pub tree: HashMap<usize, PUCTNodeInfo<G>>,
}

impl<'a, G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> PUCTPolicy<'a, G, F> {
    fn simulate(self: &mut PUCTPolicy<'a, G, F>, board: &G) {
        let mut b = board.clone();
        let history = self.select(&mut b);
        let value = if !b.is_finished() {
            let (policy, value) = (self.evaluate)(&b);
            self.expand(&b, policy);
            value
        } else {
            if b.has_won(self.color) {
                1.
            } else {
                0.
            }
        };
        self.backup(history, value);
    }

    fn select(self: &mut PUCTPolicy<'a, G, F>, b: &mut G) -> Vec<(usize, G::Move)> {
        let mut history: Vec<(usize, G::Move)> = Vec::new();

        while !b.is_finished() {
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
        max_move
            .or_else(|| panic!("Failed to select a move. {:?}", node_info.moves))
            .unwrap()
    }

    fn expand(self: &mut PUCTPolicy<'a, G, F>, board: &G, policy: HashMap<G::Move, f32>) {
        // normalize policy probabilities among possible moves.
        let z: f32 = board
            .possible_moves()
            .into_iter()
            .map(|m| policy.get(&m).unwrap())
            .sum();
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

    fn backup(self: &mut PUCTPolicy<'a, G, F>, history: Vec<(usize, G::Move)>, value: f32) {
        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (value - (*v).Q) / (*v).N_a;
        }
    }
}

impl<'a, G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> MultiplayerPolicy<G>
    for PUCTPolicy<'a, G, F>
{
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

// POLICY BUILDER

pub struct PUCT<'a, G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> {
    pub C_PUCT: f32,
    pub evaluate: &'a F,
    pub _g: PhantomData<fn() -> G>,
}

impl<G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> Copy for PUCT<'_, G, F> {}

impl<G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> Clone for PUCT<'_, G, F> {
    fn clone(&self) -> Self {
        *self
    }
}

use std::fmt;
impl<G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> fmt::Display
    for PUCT<'_, G, F>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PUCT")?;
        writeln!(f, "|| C_PUCT: {}", self.C_PUCT)?;
        writeln!(f, "|| N_PLAYOUTS: {}", N_PLAYOUTS)
    }
}

impl<'a, G: MultiplayerGame, F: Fn(&G) -> (HashMap<G::Move, f32>, f32)> MultiplayerPolicyBuilder<G>
    for PUCT<'a, G, F>
{
    type P = PUCTPolicy<'a, G, F>;

    fn create(&self, color: G::Player) -> Self::P {
        PUCTPolicy::<'a, G, F> {
            color,
            C_PUCT: self.C_PUCT,
            evaluate: self.evaluate,
            tree: HashMap::new(),
        }
    }
}
