use std::f64;
use std::iter::*;

use super::super::game::Game;
use super::{Policy, PolicyBuilder, N_PLAYOUTS};

pub struct NMCSPolicy<G: Game> {
    color: G::Player,
    s: NMCS,
}

impl<G: Game> NMCSPolicy<G> {
    fn nested(self: &NMCSPolicy<G>, board: &G, level: usize) -> (f64, Vec<Option<G::Move>>) {
        if level == 0 {
            let (board, mut history) = board.playout_board_history();
            history.reverse();
            (
                board.score(self.color),
                history.into_iter().map(|(_, b)| b).collect(),
            )
        } else {
            let mut best_score = 0.0;
            let mut best_sequence = Vec::new();
            let mut state = board.clone();
            let mut played_sequence = Vec::new();
            while board.winner() == None {
                for m in board.possible_moves() {
                    let mut new_board = board.clone();
                    new_board.play(&m);
                    let (score, history) = self.nested(&new_board, level - 1);
                    if score > best_score {
                        best_sequence = history;
                        best_sequence.push(Some(m));
                        best_score = score;
                    }
                }
                match best_sequence.pop() {
                    None => state.pass(),
                    Some(None) => {
                        state.pass();
                        played_sequence.push(None);
                    }
                    Some(Some(a)) => {
                        state.play(&a);
                        played_sequence.push(Some(a));
                    }
                }
            }
            played_sequence.reverse();
            (best_score, played_sequence)
        }
    }
}

impl<G: Game> Policy<G> for NMCSPolicy<G> {
    fn play(self: &mut NMCSPolicy<G>, board: &G) -> G::Move {
        let mut best_move = None;
        let mut max_visited = 0.;
        for m in board.possible_moves().iter() {
            let mut new_board = board.clone();
            new_board.play(&m);
            let (value, _) = self.nested(&new_board, self.s.level);

            if value >= max_visited {
                max_visited = value;
                best_move = Some(*m);
            }
        }
        best_move.unwrap()
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

impl<G: Game> PolicyBuilder<G> for NMCS {
    type P = NMCSPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        assert_eq!(G::players().len(), 1);
        NMCSPolicy { color, s: *self }
    }
}

/* MULTI NMCS  */
pub struct MultiNMCSPolicy<G: Game> {
    color: G::Player,
    s: MultiNMCS,
}

impl<G: Game> MultiNMCSPolicy<G> {
    fn nested(self: &MultiNMCSPolicy<G>, board: &G, level: usize, depth: f64, bound: f64) -> f64 {
        let mut d = depth;
        let mut s = board.clone();
        while s.winner() == None {
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

        if self.s.discounting {
            board.score(self.color) / d
        } else {
            board.score(self.color)
        }
    }
}

impl<G: Game> Policy<G> for MultiNMCSPolicy<G> {
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
    bound: f64,
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

impl<G: Game> PolicyBuilder<G> for MultiNMCS {
    type P = MultiNMCSPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        MultiNMCSPolicy { color, s: *self }
    }
}
