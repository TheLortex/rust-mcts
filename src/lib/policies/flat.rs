use rand::seq::SliceRandom;
use std::collections::HashMap;

use super::super::game::{MultiplayerGame, SingleplayerGame};
use super::{MultiplayerPolicy, MultiplayerPolicyBuilder, SingleplayerPolicy, SingleplayerPolicyBuilder, N_PLAYOUTS};

/* RANDOM POLICY */

pub struct RandomPolicy {}

impl<G: MultiplayerGame> MultiplayerPolicy<G> for RandomPolicy {
    fn play(self: &mut RandomPolicy, board: &G) -> G::Move {
        let moves = board.possible_moves();
        moves.choose(&mut rand::thread_rng()).copied().unwrap()
    }
}

impl<G: SingleplayerGame> SingleplayerPolicy<G> for RandomPolicy {
    fn solve(self: &mut RandomPolicy, board: &G) -> Vec<G::Move> {
        let b = board.clone();
        let (_, h) = b.playout_board_history();
        h.iter().map(|(_,b)| *b).collect()
    }
}

#[derive(Default)]
pub struct Random {}

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for Random {
    type P = RandomPolicy;

    fn create(&self, _: G::Player) -> Self::P {
        RandomPolicy {}
    }
}

impl<G: SingleplayerGame> SingleplayerPolicyBuilder<G> for Random {
    type P = RandomPolicy;

    fn create(&self) -> Self::P {
        RandomPolicy {}
    }
}

/* FLAT MONTE CARLO POLICY */

pub struct FlatMonteCarloPolicy<G: MultiplayerGame> {
    color: G::Player,
}

impl<G: MultiplayerGame> MultiplayerPolicy<G> for FlatMonteCarloPolicy<G> {
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
                if b_after_move.playout_board().has_won(self.color) {
                    success += 1;
                }
            }

            if success >= best_score {
                best_score = success;
                best_move = Some(m);
            }
        }

        best_move.copied().unwrap()
    }
}

#[derive(Default)]
pub struct FlatMonteCarlo {}

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for FlatMonteCarlo {
    type P = FlatMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatMonteCarloPolicy { color }
    }
}

/* Flat UCB */
pub struct FlatUCBMonteCarloPolicy<G: MultiplayerGame> {
    color: G::Player,
}

impl<G: MultiplayerGame> MultiplayerPolicy<G> for FlatUCBMonteCarloPolicy<G> {
    fn play(self: &mut FlatUCBMonteCarloPolicy<G>, board: &G) -> G::Move {
        const UCB_WEIGHT: f32 = 0.4;
        const UCB_PLAYOUTS: usize = N_PLAYOUTS;

        let moves = board.possible_moves();
        let mut move_success = HashMap::new();
        let mut move_count = HashMap::new();
        let mut move_board = HashMap::new();

        for m in moves.iter() {
            let mut b_after_move = board.clone();
            b_after_move.play(&m);
            if b_after_move.playout_board().has_won(self.color) {
                move_success.insert(m, 1);
            } else {
                move_success.insert(m, 0);
            }
            move_count.insert(m, 1);
            move_board.insert(m, b_after_move);
        }

        for i in 0..(UCB_PLAYOUTS - moves.len()) {
            let mut max_ucb = 0f32;
            let mut max_move = None;

            for m in moves.iter() {
                let count = *move_count.get(&m).unwrap() as f32;
                let succ = *move_success.get(&m).unwrap() as f32;
                let mean = succ / count;
                let ucb = mean + UCB_WEIGHT * (((moves.len() + i) as f32).ln() / count).sqrt();

                if ucb >= max_ucb {
                    max_move = Some(m);
                    max_ucb = ucb;
                }
            }

            if let Some(max_move) = max_move {
                *move_count.get_mut(&max_move).unwrap() += 1;
                if move_board.get(&max_move).unwrap().playout_board().has_won(self.color) {
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
        max_move.copied().unwrap()
    }
}

#[derive(Default)]
pub struct FlatUCBMonteCarlo {}

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for FlatUCBMonteCarlo {
    type P = FlatUCBMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatUCBMonteCarloPolicy { color }
    }
}
