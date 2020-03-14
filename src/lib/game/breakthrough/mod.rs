use ansi_term::Colour::Fixed;
use ansi_term::Style;
use rand::Rng;
use std::fmt;

use std::hash::*;

use super::{BaseGame, MultiplayerGame, InteractiveGame, MultiplayerGameBuilder, Feature};

pub mod ui;

pub const K: usize = 5;
/* PLAYERS */
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Black = 0,
    White = 1,
}

impl Color {
    pub fn adv(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    pub fn random() -> Color {
        if rand::random() {
            Color::Black
        } else {
            Color::White
        }
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = Fixed(9);
        let w = Fixed(15);

        match self {
            Color::Black => write!(f, "{}", b.paint("▓▓")),
            Color::White => write!(f, "{}", w.paint("▓▓")),
        }
    }
}

/* CELL */
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    C(Color),
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let style = Style::new().on(Fixed(0));

        match self {
            Cell::Empty => write!(f, "{}", style.paint("  ")),
            Cell::C(c) => write!(f, "{:?}", c),
        }
    }
}

/* MOVE */

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MoveDirection {
    Front,
    FrontLeft,
    FrontRight,
}

impl MoveDirection {
    fn all() -> Vec<Self> {
        vec![
            MoveDirection::FrontLeft,
            MoveDirection::FrontRight,
            MoveDirection::Front,
        ]
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Move {
    pub color: Color,
    pub x: usize,
    pub y: usize,
    pub direction: MoveDirection,
}

impl Move {
    pub fn target(&self, content: &[[Cell; K]; K]) -> (usize, usize) {
        let c = content[self.x][self.y];
        assert_ne!(c, Cell::Empty);

        let delta_y = if c == Cell::C(Color::Black) { 1 } else { -1 };
        let delta_x = match self.direction {
            MoveDirection::Front => 0,
            MoveDirection::FrontLeft => delta_y,
            MoveDirection::FrontRight => -delta_y,
        };
        let px = (self.x as i32 + delta_x) as usize;
        let py = (self.y as i32 + delta_y) as usize;
        (px, py)
    }

    pub fn is_valid(&self, content: &[[Cell; K]; K]) -> Option<(usize, usize)> {
        let c = content[self.x][self.y];
        if c != Cell::C(self.color) {
            return None;
        }
        let (px, py) = self.target(content);
        if px < K && py < K {
            if self.direction == MoveDirection::Front {
                if content[px][py] == Cell::Empty {
                    Some((px, py))
                } else {
                    None
                }
            } else if content[px][py] == Cell::Empty || content[px][py] != c {
                Some((px, py))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
pub enum PendingMove {
    SelectingPosition(usize, usize),
    SelectingMove(Move),
}

#[derive(Clone, PartialEq, Eq)]
pub struct Breakthrough {
    pub content: [[Cell; K]; K],
    possible_moves_black: Vec<Move>,
    possible_moves_white: Vec<Move>,
    transposition: [[[usize; K]; K]; 2],
    hash: usize,
    turn: Color,
}

impl fmt::Debug for Breakthrough {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let style = Style::new().on(Fixed(0));
        writeln!(f, "Turn: {:?}", self.turn)?;
        writeln!(
            f,
            "{}{}{}",
            style.paint("╔"),
            style.paint("══╤".repeat(K - 1)),
            style.paint("══╗")
        )?;
        for y in 0..K {
            if y != 0 {
                writeln!(
                    f,
                    "{}{}{}",
                    style.paint("╟"),
                    style.paint("──┼".repeat(K - 1)),
                    style.paint("──╢")
                )?;
            }
            write!(f, "{}", style.paint("║"))?;
            for x in 0..K {
                if x == 0 {
                    write!(f, "{:?}", self.content[x][y])?;
                } else {
                    write!(f, "{}{:?}", style.paint("│"), self.content[x][y])?;
                }
            }
            writeln!(f, "║")?;
        }
        writeln!(
            f,
            "{}{}{}",
            style.paint("╚"),
            style.paint("══╧".repeat(K - 1)),
            style.paint("══╝")
        )
    }
}

#[derive(Default)]
pub struct BreakthroughBuilder {}

impl MultiplayerGameBuilder<Breakthrough> for BreakthroughBuilder {
    fn create(&self, turn: Color) -> Breakthrough {
        let mut rng = rand::thread_rng();

        let mut content = [[Cell::Empty; K]; K];
        let mut transposition = [[[0; K]; K]; 2];
        for (x, column) in content.iter_mut().enumerate() {
            column[0] = Cell::C(Color::Black);
            column[1] = Cell::C(Color::Black);
            column[K - 2] = Cell::C(Color::White);
            column[K - 1] = Cell::C(Color::White);

            for y in 0..K {
                transposition[Color::Black as usize][x][y] = rng.gen::<usize>();
                transposition[Color::White as usize][x][y] = rng.gen::<usize>()
            }
        }

        let (possible_moves_black, possible_moves_white) =
            Breakthrough::compute_possible_moves(&content);

        Breakthrough {
            turn,
            content,
            possible_moves_black,
            possible_moves_white,
            transposition,
            hash: 0,
        }
    }
}

impl Breakthrough {
    pub fn winner(&self) -> Option<Color> {
        for i in 0..K {
            if self.content[i][K - 1] == Cell::C(Color::Black) {
                return Some(Color::Black);
            } else if self.content[i][0] == Cell::C(Color::White) {
                return Some(Color::White);
            }
        }
        if self.possible_moves_black.is_empty() {
            Some(Color::White)
        } else if self.possible_moves_white.is_empty() {
            Some(Color::Black)
        } else {
            None
        }
    }
}

impl MultiplayerGame for Breakthrough {
    type Player = Color;
    fn players() -> Vec<Color> {
        vec![Color::Black, Color::White]
    }
    fn turn(&self) -> Color {
        self.turn
    }

    fn has_won(&self, player: Color) -> bool {
        self.winner() == Some(player)
    }
}

impl BaseGame for Breakthrough {
    type Move = Move;

    fn play(&mut self, m: &Move) {
        if m.color != self.turn() {
            panic!("Wait. Not your turn.");
        }
        match m.is_valid(&self.content) {
            None => panic!("Wait. This is illegal. "),
            Some((px, py)) => {
                let mut c_hash = 0;
                if let Cell::C(color) = self.content[m.x][m.y] {
                    // remove cell from initial position
                    c_hash ^= self.transposition[color as usize][m.x][m.y];
                    // add cell to new position
                    c_hash ^= self.transposition[color as usize][px][py];
                }
                if let Cell::C(color) = self.content[px][py] {
                    // eat the other cell
                    c_hash ^= self.transposition[color as usize][px][py];
                }
                self.hash ^= c_hash;
                assert_eq!(self.content[m.x][m.y], Cell::C(self.turn));
                assert_ne!(self.content[px][py], Cell::C(self.turn));

                self.content[px][py] = self.content[m.x][m.y];
                self.content[m.x][m.y] = Cell::Empty;

                // update possible moves state
                let (pmb, pmw) = Breakthrough::compute_possible_moves(&self.content);
                self.possible_moves_black = pmb;
                self.possible_moves_white = pmw;

                self.turn = self.turn.adv();
            }
        }
    }


    fn hash(&self) -> usize {
        (self.hash << 1) + (self.turn as usize)
    }

    fn possible_moves(&self) -> &[Move] {
        if self.turn == Color::Black {
            &self.possible_moves_black
        } else {
            &self.possible_moves_white
        }
    }

    fn is_finished(&self) -> bool {
        self.winner().is_some()
    }
}

impl Breakthrough {
    pub fn show(&self) {
        println!("{:?}", self);
    }

    fn compute_possible_moves(content: &[[Cell; K]; K]) -> (Vec<Move>, Vec<Move>) {
        let mut result_black = vec![];
        let mut result_white = vec![];

        for x in 0..K {
            for y in 0..K {
                if let Cell::C(color) = content[x][y] {
                    for direction in &[
                        MoveDirection::Front,
                        MoveDirection::FrontLeft,
                        MoveDirection::FrontRight,
                    ] {
                        let m = Move {
                            color,
                            x,
                            y,
                            direction: *direction,
                        };
                        if m.is_valid(content).is_some() {
                            if color == Color::Black {
                                result_black.push(m)
                            } else {
                                result_white.push(m)
                            }
                        }
                    }
                }
            }
        }
        (result_black, result_white)
    }
}


use ndarray::{Array};
use std::collections::HashMap;
use std::iter::FromIterator;

impl Feature for Breakthrough {
    type StateDim = ndarray::Ix3;
    type ActionDim = ndarray::Ix3;

    fn state_dimension() -> Self::StateDim {
        ndarray::Dim([K, K, 3])
    }

    fn action_dimension() -> Self::ActionDim{
        ndarray::Dim([K, K, 3])
    }

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim> {
        let mut features = ndarray::Array::zeros(Self::state_dimension());

        for ((x,y,z), row) in features.indexed_iter_mut() {
            if z == 0 && self.content[x][y] == Cell::C(pov) {
                *row = 1.0
            } else if z == 1 && self.content[x][y] == Cell::C(pov.adv()) {
                *row = 1.0
            } else if z == 2 && self.turn() == pov {
                *row = 1.0
            }
        }
        
        features
    }

    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim> {
        let mut features = ndarray::Array::zeros(Self::action_dimension());

        for (action, proba) in moves.iter() {
            features[[action.x, action.y, action.direction as usize]] = *proba;
        }

        features
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        let z: f32 = self.possible_moves().iter().map(|m| features[[m.x, m.y, m.direction as usize]]).sum();
        HashMap::from_iter(
            self.possible_moves().iter().map(|m| (*m, features[[m.x, m.y, m.direction as usize]] / z))
        )
    }
}