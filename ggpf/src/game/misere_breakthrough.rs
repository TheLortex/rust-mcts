use crate::game::breakthrough::{Breakthrough, BreakthroughBuilder, Color, Move};
use crate::game::{Base, Game, GameBuilder, Playable};

use async_trait::async_trait;
use std::fmt;

/// Misère breakthrough
#[derive(Clone, PartialEq, Eq)]
pub struct MisereBreakthrough {
    game: Breakthrough,
}

impl fmt::Debug for MisereBreakthrough {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.game.fmt(f)
    }
}

#[derive(Clone)]
struct Misere(BreakthroughBuilder);

#[async_trait]
impl GameBuilder for Misere {
    type G = MisereBreakthrough;

    async fn create(&self, turn: Color) -> MisereBreakthrough {
        MisereBreakthrough {
            game: self.0.create(turn).await,
        }
    }
}

impl Base for MisereBreakthrough {
    type Move = Move;

    fn possible_moves(&self) -> Vec<Move> {
        self.game.possible_moves()
    }
}

#[async_trait]
impl Playable for MisereBreakthrough {
    async fn play(&mut self, m: &Move) -> f32 {
        -self.game.play(m).await
    }
}

impl Game for MisereBreakthrough {
    type Player = Color;

    fn players() -> Vec<Color> {
        Breakthrough::players()
    }

    fn turn(&self) -> Color {
        self.game.turn()
    }

    fn player_after(player: Self::Player) -> Self::Player {
        Breakthrough::player_after(player)
    }
}

impl MisereBreakthrough {
    /// Show game state
    pub fn show(&self) {
        self.game.show()
    }
}
