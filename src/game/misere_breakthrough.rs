use std::fmt;

use super::Game;
use super::breakthrough::{Breakthrough, Color, Move};

const K: usize = 5;

#[derive(Clone, PartialEq, Eq)]
pub struct MisereBreakthrough {
    game: Breakthrough,
}

impl fmt::Debug for MisereBreakthrough {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.game.fmt(f)
    }
}

impl Game for MisereBreakthrough {
    type Player = Color;
    type Move = Move;

    type GameHash = usize;
    type Settings = ();

    fn new(turn: Color, _: ()) -> MisereBreakthrough {
        MisereBreakthrough {
            game: Breakthrough::new(turn, ())
        }
    }

    fn players() -> Vec<Color> {
        Breakthrough::players()
    }

    fn play(&mut self, m: &Move) {
        self.game.play(m)
    }

    fn turn(&self) -> Color {
        self.game.turn()
    }

    fn hash(&self) -> usize {
        self.game.hash()
    }

    fn possible_moves(&self) -> &Vec<Move> {
        self.game.possible_moves()
    }

    fn winner(&self) -> Option<Color> {
        self.game.winner().map(|c| c.adv())
    }

    fn score(&self, c: Self::Player) -> f32 {
        match self.winner() {
            Some (c_) if c == c_ => 1.,
            Some (_) => -1.,
            _ => 0.,
        }
    }

    fn pass(&mut self) {
        self.game.pass()
    }
}

impl MisereBreakthrough {
    pub fn show(&self) {
        self.game.show()
    }
}
