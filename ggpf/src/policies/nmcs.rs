use crate::game::*;
use crate::policies::{
    MultiplayerPolicy, MultiplayerPolicyBuilder, SingleplayerPolicy, SingleplayerPolicyBuilder,
};

use async_trait::async_trait;
use futures::future::{BoxFuture, FutureExt};
use std::f32;
use std::iter::*;

/// Nested Monte Carlo Search
pub struct NMCSPolicy {
    s: NMCS,
}

impl NMCSPolicy {
    fn nested<'a, G: Singleplayer + Clone>(
        self: &'a NMCSPolicy,
        board: &'a G,
        level: usize,
    ) -> BoxFuture<'a, (f32, Vec<G::Move>)> {
        async move {
            if level == 0 {
                let (_board, mut history, total_reward) = board.playout_history(0).await;
                history.reverse();
                (total_reward, history.into_iter().map(|(_, b)| b).collect())
            } else {
                let mut best_score = 0.0;
                let mut best_sequence = Vec::new();
                let mut state = board.clone();
                let mut played_sequence = Vec::new();

                while !state.is_finished() {
                    for m in state.possible_moves() {
                        let mut new_board = state.clone();
                        new_board.play(&m).await;
                        let (score, history) = self.nested(&new_board, level - 1).await;
                        if score >= best_score {
                            best_sequence = history;
                            best_sequence.push(m);
                            best_score = score;
                        }
                    }
                    let next_action = best_sequence.pop().unwrap();
                    state.play(&next_action).await;
                    played_sequence.push(next_action);
                }
                played_sequence.reverse();
                (best_score, played_sequence)
            }
        }
        .boxed()
    }
}

#[async_trait]
impl<G: Singleplayer + Clone> SingleplayerPolicy<G> for NMCSPolicy {
    async fn solve(self: &mut NMCSPolicy, board: &G) -> Vec<G::Move> {
        let (_, mut sequence) = self.nested(board, self.s.level).await;
        sequence.reverse();
        sequence
    }
}

/// Nested Monte Carlo Search policy builder.
#[derive(Copy, Clone)]
pub struct NMCS {
    level: usize,
}

impl Default for NMCS {
    fn default() -> NMCS {
        NMCS { level: 2 }
    }
}

impl<G: Singleplayer + Clone> SingleplayerPolicyBuilder<G> for NMCS {
    type P = NMCSPolicy;

    fn create(&self) -> Self::P {
        NMCSPolicy { s: *self }
    }
}

/// Multiplayer NMCS
pub struct MultiNMCSPolicy<G: Game> {
    color: G::Player,
    s: MultiNMCS,
}

impl<G: Game + SingleWinner + Clone> MultiNMCSPolicy<G> {
    fn nested<'a>(
        self: &'a MultiNMCSPolicy<G>,
        board: &'a G,
        level: usize,
        depth: f32,
        bound: f32,
    ) -> BoxFuture<'a, f32> {
        async move {
            let mut d = depth;
            let mut s = board.clone();

            while !s.is_finished() {
                let mut s_star = s.clone();
                s_star.random_move().await;
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
                        new_s.play(&m).await;
                        let l = self.nested(&new_s, level - 1, d + 1., l_star).await;
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

            let score = if board.winner() == Some(self.color) {
                1.
            } else if !board.is_finished() {
                0.0
            } else {
                -1.0
            };

            if self.s.discounting {
                score / d
            } else {
                score
            }
        }
        .boxed()
    }
}

#[async_trait]
impl<G: Game + SingleWinner + Clone> MultiplayerPolicy<G> for MultiNMCSPolicy<G> {
    async fn play(self: &mut MultiNMCSPolicy<G>, board: &G) -> G::Move {
        let mut best_move = None;
        let mut max_visited = 0.;

        for m in board.possible_moves() {
            let mut new_board = board.clone();
            new_board.play(&m).await;
            let value = self
                .nested(&new_board, self.s.level, 0., self.s.bound)
                .await;

            if value >= max_visited {
                max_visited = value;
                best_move = Some(m);
            }
        }
        best_move.unwrap()
    }
}

/// Multiplayer NMCS policy builder.
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
            level: 3,
            bound: 1.0,
        }
    }
}

use std::fmt;
impl fmt::Display for MultiNMCS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MultiNMCS")?;
        writeln!(f, "|| Discounting: {}", self.discounting)?;
        writeln!(f, "|| D_pruning: {}", self.d_pruning)?;
        writeln!(f, "|| Cut_on_win: {}", self.cut_on_win)?;
        writeln!(f, "|| LEVEL: {}", self.level)?;
        writeln!(f, "|| BOUND: {}", self.bound)
    }
}

impl<G: Game + SingleWinner + Clone> MultiplayerPolicyBuilder<G> for MultiNMCS {
    type P = MultiNMCSPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        MultiNMCSPolicy { color, s: *self }
    }
}
