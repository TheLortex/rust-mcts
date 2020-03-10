use std::f32;
use std::iter::*;

use super::super::game::{SingleplayerGame, MultiplayerGame};
use super::{MultiplayerPolicy, MultiplayerPolicyBuilder, SingleplayerPolicy, SingleplayerPolicyBuilder};

pub struct NMCSPolicy {
    s: NMCS,
}

impl NMCSPolicy {
    fn nested<G: SingleplayerGame>(self: &NMCSPolicy, board: &G, level: usize) -> (f32, Vec<G::Move>) {
        if level == 0 {
            let (board, mut history) = board.playout_board_history();
            history.reverse();
            (
                board.score(),
                history.into_iter().map(|(_, b)| b).collect(),
            )
        } else {
            let mut best_score = 0.0;
            let mut best_sequence = Vec::new();
            let mut state = board.clone();
            let mut played_sequence = Vec::new();

            while !state.is_finished() {
                for m in state.possible_moves() {
                    let mut new_board = state.clone();
                    new_board.play(m);
                    let (score, history) = self.nested(&new_board, level - 1);
                    if score >= best_score {
                        best_sequence = history;
                        best_sequence.push(*m);
                        best_score = score;
                    }
                }
                let next_action = best_sequence.pop().unwrap();
                state.play(&next_action);
                played_sequence.push(next_action);
            }
            played_sequence.reverse();
            (best_score, played_sequence)
        }
    }
}

impl<G: SingleplayerGame> SingleplayerPolicy<G> for NMCSPolicy {
    fn solve(self: &mut NMCSPolicy, board: &G) -> Vec<G::Move> {
        let (_, mut sequence) = self.nested(board, self.s.level);
        sequence.reverse();
        sequence
    }
}

#[derive(Copy, Clone)]
pub struct NMCS {
    level: usize,
}

impl Default for NMCS {
    fn default() -> NMCS {
        NMCS { level: 2 }
    }
}

impl<G: SingleplayerGame> SingleplayerPolicyBuilder<G> for NMCS {
    type P = NMCSPolicy;

    fn create(&self) -> Self::P {
        NMCSPolicy { s: *self }
    }
}

/* MULTI NMCS  */
pub struct MultiNMCSPolicy<G: MultiplayerGame> {
    color: G::Player,
    s: MultiNMCS,
}

impl<G: MultiplayerGame> MultiNMCSPolicy<G> {
    fn nested(self: &MultiNMCSPolicy<G>, board: &G, level: usize, depth: f32, bound: f32) -> f32 {
        let mut d = depth;
        let mut s = board.clone();
        while !s.is_finished() {
            let mut s_star = s.clone();
            s_star.random_move();
            let mut l_star = if s.turn() == self.color {
                -1. / d
            } else {
                1. / d
            };

            if self.s.d_pruning
                && ((s.turn() == self.color && -l_star < bound)
                    || (s.turn() != self.color && -l_star > bound))
            {
                return bound;
            }

            if depth > 0. {
                for m in s.possible_moves() {
                    let mut new_s = s.clone();
                    new_s.play(&m);
                    let l = self.nested(&new_s, level - 1, d + 1., l_star);
                    if (s.turn() == self.color && l > l_star)
                        || (s.turn() != self.color && l < l_star)
                    {
                        l_star = l;
                        s_star = new_s;
                    }
                    if self.s.cut_on_win && (l != 0.) {
                        break;
                    }
                }
            }

            s = s_star;
            d += 1.;
        }

        let score = if board.has_won(self.color) { 1. } else if !board.is_finished() { 0.0 } else { -1.0};

        if self.s.discounting {
            score / d
        } else {
            score
        }
    }
}

impl<G: MultiplayerGame> MultiplayerPolicy<G> for MultiNMCSPolicy<G> {
    fn play(self: &mut MultiNMCSPolicy<G>, board: &G) -> G::Move {
        let mut best_move = None;
        let mut max_visited = 0.;
        for m in board.possible_moves().iter() {
            let mut new_board = board.clone();
            new_board.play(&m);
            let value = self.nested(&new_board, self.s.level, 0., self.s.bound);

            if value >= max_visited {
                max_visited = value;
                best_move = Some(*m);
            }
        }
        best_move.unwrap()
    }
}

#[derive(Copy, Clone)]
pub struct MultiNMCS {
    discounting: bool,
    d_pruning: bool,
    cut_on_win: bool,
    level: usize,
    bound: f32,
}

impl Default for MultiNMCS {
    fn default() -> MultiNMCS {
        MultiNMCS {
            discounting: true,
            d_pruning: true,
            cut_on_win: true,
            level: 2,
            bound: 1.0,
        }
    }
}

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for MultiNMCS {
    type P = MultiNMCSPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        MultiNMCSPolicy { color, s: *self }
    }
}
