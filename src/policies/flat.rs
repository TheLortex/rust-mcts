use rand::seq::SliceRandom;
use std::collections::HashMap;

use super::super::game::Game;
use super::{Policy, PolicyBuilder, N_PLAYOUTS};

/* RANDOM POLICY */

pub struct RandomPolicy<G: Game> {
    color: G::Player,
}

impl<G: Game> Policy<G> for RandomPolicy<G> {
    fn play(self: &mut RandomPolicy<G>, board: &G) -> G::Move {
        let moves = board.possible_moves();
        moves.choose(&mut rand::thread_rng()).map(|x| *x).unwrap()
    }
}

#[derive(Default)]
pub struct Random {}

impl<G: Game> PolicyBuilder<G> for Random {
    type P = RandomPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        RandomPolicy { color }
    }
}

/* FLAT MONTE CARLO POLICY */

pub struct FlatMonteCarloPolicy<G: Game> {
    color: G::Player,
}

impl<G: Game> Policy<G> for FlatMonteCarloPolicy<G> {
    fn play(self: &mut FlatMonteCarloPolicy<G>, board: &G) -> G::Move {
        const FLAT_PLAYOUTS: usize = N_PLAYOUTS;

        let moves = board.possible_moves();

        let n_playouts_per_move = FLAT_PLAYOUTS / moves.len();

        let mut best_move = None;
        let mut best_score = 0;

        for m in moves.iter() {
            let mut b_after_move = board.clone();
            b_after_move.play(&m);
            let mut success = 0;
            for _ in 0..n_playouts_per_move {
                if b_after_move.playout() == self.color {
                    success += 1;
                }
            }

            if success >= best_score {
                best_score = success;
                best_move = Some(m);
            }
        }

        best_move.map(|x| *x).unwrap()
    }
}

pub struct FlatMonteCarlo {}

impl<G: Game> PolicyBuilder<G> for FlatMonteCarlo {
    type P = FlatMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatMonteCarloPolicy { color }
    }
}

/* Flat UCB */
pub struct FlatUCBMonteCarloPolicy<G: Game> {
    color: G::Player,
}

impl<G: Game> Policy<G> for FlatUCBMonteCarloPolicy<G> {
    fn play(self: &mut FlatUCBMonteCarloPolicy<G>, board: &G) -> G::Move {
        const UCB_WEIGHT: f64 = 0.4;
        const UCB_PLAYOUTS: usize = N_PLAYOUTS;

        let moves = board.possible_moves();
        let mut move_success = HashMap::new();
        let mut move_count = HashMap::new();
        let mut move_board = HashMap::new();

        for m in moves.iter() {
            let mut b_after_move = board.clone();
            b_after_move.play(&m);
            if b_after_move.playout() == self.color {
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
                if move_board.get(&max_move).unwrap().playout() == self.color {
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
        max_move.map(|x| *x).unwrap()
    }
}

pub struct FlatUCBMonteCarlo {}

impl<G: Game> PolicyBuilder<G> for FlatUCBMonteCarlo {
    type P = FlatUCBMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatUCBMonteCarloPolicy { color }
    }
}
