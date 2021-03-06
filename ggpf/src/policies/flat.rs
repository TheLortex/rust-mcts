use crate::game::{Game, Playout, SingleWinner, Singleplayer};
use crate::policies::{
    MultiplayerPolicy, MultiplayerPolicyBuilder, SingleplayerPolicy, SingleplayerPolicyBuilder,
};
use crate::settings;

use async_trait::async_trait;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Random policy
///
/// Takes a random move at each step.
pub struct RandomPolicy {}

#[async_trait]
impl<G: Game> MultiplayerPolicy<G> for RandomPolicy {
    async fn play(self: &mut RandomPolicy, board: &G) -> G::Move {
        let moves = board.possible_moves();
        moves.choose(&mut rand::thread_rng()).copied().unwrap()
    }
}

#[async_trait]
impl<G: Singleplayer + Clone> SingleplayerPolicy<G> for RandomPolicy {
    async fn solve(self: &mut RandomPolicy, board: &G) -> Vec<G::Move> {
        let b = board.clone();
        let (_, h, _) = b.playout_history(0).await;
        h.iter().map(|(_, b)| *b).collect()
    }
}

/// Random policy builder.
#[derive(Default)]
pub struct Random {}

use std::fmt;
impl fmt::Display for Random {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Random")
    }
}

impl<G: Game> MultiplayerPolicyBuilder<G> for Random {
    type P = RandomPolicy;

    fn create(&self, _: G::Player) -> Self::P {
        RandomPolicy {}
    }
}

impl<G: Singleplayer + Clone> SingleplayerPolicyBuilder<G> for Random {
    type P = RandomPolicy;

    fn create(&self) -> Self::P {
        RandomPolicy {}
    }
}

/// Flat Monte Carlo policy
pub struct FlatMonteCarloPolicy<G: Game> {
    color: G::Player,
    playouts: usize,
}

#[async_trait]
impl<G: Game + SingleWinner + Clone> MultiplayerPolicy<G> for FlatMonteCarloPolicy<G> {
    async fn play(self: &mut FlatMonteCarloPolicy<G>, board: &G) -> G::Move {
        let moves = board.possible_moves();

        let n_playouts_per_move = self.playouts / moves.len();

        let mut best_move = None;
        let mut best_score = 0;

        for m in moves.into_iter() {
            let mut b_after_move = board.clone();
            b_after_move.play(&m).await;
            let mut success = 0;
            for _ in 0..n_playouts_per_move {
                if b_after_move.playout_board(self.color).await.0.winner() == Some(self.color) {
                    success += 1;
                }
            }

            if success >= best_score {
                best_score = success;
                best_move = Some(m);
            }
        }

        best_move.unwrap()
    }
}

/// Flat Monte Carlo policy builder
type FlatMonteCarlo = settings::FlatMonteCarlo;

impl fmt::Display for FlatMonteCarlo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FlatMonteCarlo")
    }
}

impl<G: Game + SingleWinner + Clone> MultiplayerPolicyBuilder<G> for FlatMonteCarlo {
    type P = FlatMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatMonteCarloPolicy {
            color,
            playouts: self.playouts,
        }
    }
}

/// Flat Monte Carlo with UCB policy
pub struct FlatUCBMonteCarloPolicy<G: Game> {
    color: G::Player,
    playouts: usize,
    ucb_weight: f32,
}

#[async_trait]
impl<G: Game + SingleWinner + Clone> MultiplayerPolicy<G> for FlatUCBMonteCarloPolicy<G> {
    async fn play(self: &mut FlatUCBMonteCarloPolicy<G>, board: &G) -> G::Move {
        let moves = board.possible_moves();
        let n_moves = moves.len();

        let mut move_success: HashMap<&G::Move, i32> = HashMap::new();
        let mut move_count: HashMap<&G::Move, i32> = HashMap::new();
        let mut move_board: HashMap<&G::Move, G> = HashMap::new();

        for m in moves.iter() {
            let mut b_after_move = board.clone();
            b_after_move.play(&m).await;
            if b_after_move.playout_board(self.color).await.0.winner() == Some(self.color) {
                move_success.insert(m, 1);
            } else {
                move_success.insert(m, 0);
            }
            move_count.insert(m, 1);
            move_board.insert(m, b_after_move);
        }

        for i in 0..(self.playouts - n_moves) {
            let mut max_ucb = 0f32;
            let mut max_move = None;

            for m in moves.iter() {
                let count = *move_count.get(&m).unwrap() as f32;
                let succ = *move_success.get(&m).unwrap() as f32;
                let mean = succ / count;
                let ucb = mean + self.ucb_weight * (((n_moves + i) as f32).ln() / count).sqrt();

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
                    .playout_board(self.color)
                    .await
                    .0
                    .winner()
                    == Some(self.color)
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
        max_move.copied().unwrap()
    }
}

/// Flat Monte Carlo with UCB policy builder

type FlatUCBMonteCarlo = settings::FlatUCBMonteCarlo;

impl fmt::Display for FlatUCBMonteCarlo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FlatUCBMonteCarlo")
    }
}

impl<G: Game + SingleWinner + Clone> MultiplayerPolicyBuilder<G> for FlatUCBMonteCarlo {
    type P = FlatUCBMonteCarloPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        FlatUCBMonteCarloPolicy {
            color,
            playouts: self.playouts,
            ucb_weight: self.ucb_weight,
        }
    }
}
