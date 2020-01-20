use rand::seq::SliceRandom;
use std::collections;
use std::f64;

use super::game::*;

const N_PLAYOUTS: usize = 100;
pub trait Policy {
    fn new(color: Color) -> Self;
    fn play(&self, board: &Board) -> Option<Move>;
}

pub struct RandomPolicy {
    color: Color,
}

impl Policy for RandomPolicy {
    fn new(color: Color) -> RandomPolicy {
        RandomPolicy { color }
    }

    fn play(self: &RandomPolicy, board: &Board) -> Option<Move> {
        let moves = board.possible_moves(self.color);
        moves.choose(&mut rand::thread_rng()).map(|x| *x)
    }
}

pub struct FlatMonteCarloPolicy {
    color: Color,
}

impl Policy for FlatMonteCarloPolicy {
    fn new(color: Color) -> FlatMonteCarloPolicy {
        FlatMonteCarloPolicy { color }
    }

    fn play(self: &FlatMonteCarloPolicy, board: &Board) -> Option<Move> {
        const FLAT_PLAYOUTS: usize = N_PLAYOUTS;

        let moves = board.possible_moves(self.color);
        let n_playouts_per_move = FLAT_PLAYOUTS / moves.len();

        let mut best_move = None;
        let mut best_score = 0;

        for m in moves.iter() {
            let mut b_after_move = *board;
            b_after_move.play(&m);
            let mut success = 0;
            for _ in 0..n_playouts_per_move {
                if b_after_move.playout(self.color.adv(), false) == self.color {
                    success += 1;
                }
            }

            if success >= best_score {
                best_score = success;
                best_move = Some(m);
            }
        }

        best_move.map(|x| *x)
    }
}

pub struct FlatUCBMonteCarloPolicy {
    color: Color,
}

impl Policy for FlatUCBMonteCarloPolicy {
    fn new(color: Color) -> FlatUCBMonteCarloPolicy {
        FlatUCBMonteCarloPolicy { color }
    }

    fn play(self: &FlatUCBMonteCarloPolicy, board: &Board) -> Option<Move> {
        const UCB_WEIGHT: f64 = 0.4;
        const UCB_PLAYOUTS: usize = N_PLAYOUTS;

        let moves = board.possible_moves(self.color);
        let mut move_success = collections::HashMap::new();
        let mut move_count = collections::HashMap::new();
        let mut move_board = collections::HashMap::new();

        for m in moves.iter() {
            let mut b_after_move = *board;
            b_after_move.play(&m);
            if b_after_move.playout(self.color.adv(), false) == self.color {
                move_success.insert(m, 1);
            } else {
                move_success.insert(m, 0);
            }
            move_count.insert(m, 1);
            move_board.insert(m, b_after_move);
        }

        for i in 0..(UCB_PLAYOUTS - moves.len()) {
            let mut max_ucb = 0f64;
            let mut max_move = None;

            for m in moves.iter() {
                let count = *move_count.get(&m).unwrap() as f64;
                let succ = *move_success.get(&m).unwrap() as f64;
                let mean = succ / count;
                let ucb = mean + UCB_WEIGHT * (((moves.len() + i) as f64).ln() / count).sqrt();

                if ucb >= max_ucb {
                    max_move = Some(m);
                    max_ucb = ucb;
                }
            }

            if let Some(max_move) = max_move {
                *move_count.get_mut(&max_move).unwrap() += 1;
                if move_board
                    .get(&max_move)
                    .unwrap()
                    .playout(self.color.adv(), false)
                    == self.color
                {
                    *move_success.get_mut(&max_move).unwrap() += 1;
                }
            }
        }

        let mut max_count = 0;
        let mut max_move = None;

        for m in moves.iter() {
            let count = *move_count.get(&m).unwrap();

            if count >= max_count {
                max_move = Some(m);
                max_count = count;
            }
        }
        max_move.map(|x| *x)
    }
}
