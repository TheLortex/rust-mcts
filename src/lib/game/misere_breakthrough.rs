use crate::game::{BaseGame,MultiplayerGame,MultiplayerGameBuilder};
use crate::game::breakthrough::{Breakthrough, BreakthroughBuilder, Color, Move};

use std::fmt;

#[derive(Clone, PartialEq, Eq)]
pub struct MisereBreakthrough {
    game: Breakthrough,
}

impl fmt::Debug for MisereBreakthrough {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.game.fmt(f)
    }
}

impl MultiplayerGameBuilder<MisereBreakthrough> for BreakthroughBuilder {
    fn create(&self, turn: Color) -> MisereBreakthrough {
        MisereBreakthrough {
            game: self.create(turn)
        }
    }
}

impl BaseGame for MisereBreakthrough {
    type Move = Move;

    fn play(&mut self, m: &Move) {
        self.game.play(m)
    }

    fn hash(&self) -> usize {
        self.game.hash()
    }

    fn possible_moves(&self) -> &[Move] {
        self.game.possible_moves()
    }

}

impl MultiplayerGame for MisereBreakthrough {
    type Player = Color;

    fn players() -> Vec<Color> {
        Breakthrough::players()
    }

    fn turn(&self) -> Color {
        self.game.turn()
    }

    fn has_won(&self, c: Color) -> bool {
        self.game.winner().map(|c| c.adv()) == Some(c)
    }
}

impl MisereBreakthrough {
    pub fn show(&self) {
        self.game.show()
    }
}
