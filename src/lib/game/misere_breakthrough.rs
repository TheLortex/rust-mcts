use crate::game::{Base,Game,GameBuilder};
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

impl GameBuilder<MisereBreakthrough> for BreakthroughBuilder {
    fn create(&self, turn: Color) -> MisereBreakthrough {
        MisereBreakthrough {
            game: self.create(turn)
        }
    }
}

impl Base for MisereBreakthrough {
    type Move = Move;
    type MoveIterator<'a> = <Breakthrough as Base>::MoveIterator<'a>;

    fn play(&mut self, m: &Move) {
        self.game.play(m)
    }

    fn hash(&self) -> usize {
        self.game.hash()
    }

    fn possible_moves<'a>(&'a self) -> Self::MoveIterator<'a> {
        self.game.possible_moves()
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

    fn has_won(&self, c: Color) -> bool {
        self.game.winner().map(|c| c.adv()) == Some(c)
    }
}

impl MisereBreakthrough {
    pub fn show(&self) {
        self.game.show()
    }
}
