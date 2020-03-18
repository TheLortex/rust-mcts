use rand::seq::SliceRandom;
use std::collections::HashMap;

use crate::game::{MultiplayerGame, SingleplayerGame};
use crate::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder, SingleplayerPolicy, SingleplayerPolicyBuilder};
use crate::settings;

/* RANDOM POLICY */

pub struct RandomPolicy {}

impl<G: MultiplayerGame> MultiplayerPolicy<G> for RandomPolicy {
    fn play(self: &mut RandomPolicy, history: &[G]) -> G::Move {
        let board = history.last().unwrap();
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

use std::fmt;
impl fmt::Display for Random {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Random")
    }
} 

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
    N_PLAYOUTS: usize,
}

impl<G: MultiplayerGame> MultiplayerPolicy<G> for FlatMonteCarloPolicy<G> {
    fn play(self: &mut FlatMonteCarloPolicy<G>, history: &[G]) -> G::Move {
        let board = history.last().unwrap();

        let moves = board.possible_moves();

        let n_playouts_per_move = self.N_PLAYOUTS / moves.len();

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

pub struct FlatMonteCarlo {
    N_PLAYOUTS: usize
}

impl Default for FlatMonteCarlo {
    fn default() -> Self {
        Self {
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS
        }
    }
}

impl fmt::Display for FlatMonteCarlo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FlatMonteCarlo")
    }
} 

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for FlatMonteCarlo {
    type P = FlatMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatMonteCarloPolicy { color, N_PLAYOUTS: self.N_PLAYOUTS }
    }
}

/* Flat UCB */
pub struct FlatUCBMonteCarloPolicy<G: MultiplayerGame> {
    color: G::Player,
    N_PLAYOUTS: usize,
    UCB_WEIGHT: f32,
}

impl<G: MultiplayerGame> MultiplayerPolicy<G> for FlatUCBMonteCarloPolicy<G> {
    fn play(self: &mut FlatUCBMonteCarloPolicy<G>, history: &[G]) -> G::Move {
        let board = history.last().unwrap();

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

        for i in 0..(self.N_PLAYOUTS - moves.len()) {
            let mut max_ucb = 0f32;
            let mut max_move = None;

            for m in moves.iter() {
                let count = *move_count.get(&m).unwrap() as f32;
                let succ = *move_success.get(&m).unwrap() as f32;
                let mean = succ / count;
                let ucb = mean + self.UCB_WEIGHT * (((moves.len() + i) as f32).ln() / count).sqrt();

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

pub struct FlatUCBMonteCarlo {    
    N_PLAYOUTS: usize,
    UCB_WEIGHT: f32,
}

impl Default for FlatUCBMonteCarlo {
    fn default() -> Self {
        Self {
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
            UCB_WEIGHT: 0.4,
        }
    }
}

impl fmt::Display for FlatUCBMonteCarlo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FlatUCBMonteCarlo")
    }
} 

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for FlatUCBMonteCarlo {
    type P = FlatUCBMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatUCBMonteCarloPolicy { color, N_PLAYOUTS: self.N_PLAYOUTS, UCB_WEIGHT: self.UCB_WEIGHT }
    }
}
