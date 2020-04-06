use crate::game::*;

use ansi_term::Colour::Fixed;
use ansi_term::Style;
use async_trait::async_trait;
use ndarray::{Array, ArrayView, Axis, Ix2};
use rand::Rng;
use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;

/// Breakthrough interactive interface.
pub mod ui;
/// Players
///
/// Two colors: black and white.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    /// Black
    Black = 0,
    /// White
    White = 1,
}

impl Into<u8> for Color {
    fn into(self) -> u8 {
        self as u8
    }
}

impl Color {
    /// Returns the adversary of the player.
    pub fn adv(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    /// Returns a random player.
    pub fn random() -> Color {
        if rand::random() {
            Color::Black
        } else {
            Color::White
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = Fixed(9);
        let w = Fixed(15);

        match self {
            Color::Black => write!(f, "{}", b.paint("▓▓")),
            Color::White => write!(f, "{}", w.paint("▓▓")),
        }
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::Black => write!(f, "B"),
            Color::White => write!(f, "W"),
        }
    }
}

/// Game cell
///
/// Represents a position on the board.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// Empty cell.
    Empty,
    /// Cell containing a pawn of given color.
    C(Color),
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let style = Style::new().on(Fixed(0));

        match self {
            Cell::Empty => write!(f, "{}", style.paint("  ")),
            Cell::C(c) => write!(f, "{}", c),
        }
    }
}

/// Move direction
///
/// Possible move directions relative to the pawn position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MoveDirection {
    /// Front
    Front,
    /// Front left
    FrontLeft,
    /// Front right
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

/// Move
///
/// Describes a potentially legal action on the board.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct Move {
    /// Player
    pub color: Color,
    /// Current position x
    pub x: usize,
    /// Current position y
    pub y: usize,
    /// Move direction
    pub direction: MoveDirection,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Move {
    /// Write a human readable name for the move.
    pub fn name(&self) -> String {
        let (px, py) = self.target(); // todo: extract helper
        format!(
            "{:?} {}{}->{}{}",
            self.color,
            ('a' as usize + self.x) as u8 as char,
            1 + self.y,
            ('a' as usize + px) as u8 as char,
            1 + py
        )
    }

    /// Compute move target.
    pub fn target(&self) -> (usize, usize) {
        let delta_y = if self.color == Color::Black { 1 } else { -1 };
        let delta_x = match self.direction {
            MoveDirection::Front => 0,
            MoveDirection::FrontLeft => delta_y,
            MoveDirection::FrontRight => -delta_y,
        };
        let px = (self.x as i32 + delta_x) as usize;
        let py = (self.y as i32 + delta_y) as usize;
        (px, py)
    }

    /// Check if move is valid on the given board.
    ///
    /// Returns the target coordinate in this case.
    pub fn is_valid(&self, content: ArrayView<Cell, Ix2>) -> Option<(usize, usize)> {
        let c = content[[self.x, self.y]];
        if c != Cell::C(self.color) {
            return None;
        }
        let K = content.len_of(Axis(0));
        let (px, py) = self.target();
        if px < K && py < K {
            if self.direction == MoveDirection::Front {
                if content[[px, py]] == Cell::Empty {
                    Some((px, py))
                } else {
                    None
                }
            } else if content[[px, py]] == Cell::Empty || content[[px, py]] != c {
                Some((px, py))
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Breakthrough game state instance
#[derive(Clone, Eq)]
pub struct Breakthrough {
    K: usize,
    content: ndarray::Array2<Cell>,

    transposition: ndarray::Array3<usize>,
    hash: usize,
    turn: Color,
}

impl PartialEq for Breakthrough {
    fn eq(&self, other: &Self) -> bool {
        self.content.eq(&other.content) && self.turn == other.turn
    }
}

impl Hash for Breakthrough {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.transposition.hash(state)
    }
}

impl fmt::Debug for Breakthrough {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let style = Style::new().on(Fixed(0));
        writeln!(f, "Turn: {:?}", self.turn)?;
        writeln!(
            f,
            "{}{}{}",
            style.paint("╔"),
            style.paint("══╤".repeat(self.K - 1)),
            style.paint("══╗")
        )?;
        for y in 0..self.K {
            if y != 0 {
                writeln!(
                    f,
                    "{}{}{}",
                    style.paint("╟"),
                    style.paint("──┼".repeat(self.K - 1)),
                    style.paint("──╢")
                )?;
            }
            write!(f, "{}", style.paint("║"))?;
            for x in 0..self.K {
                if x == 0 {
                    write!(f, "{:?}", self.content[[x, y]])?;
                } else {
                    write!(f, "{}{:?}", style.paint("│"), self.content[[x, y]])?;
                }
            }
            writeln!(f, "║")?;
        }
        writeln!(
            f,
            "{}{}{}",
            style.paint("╚"),
            style.paint("══╧".repeat(self.K - 1)),
            style.paint("══╝")
        )
    }
}

/// Game builder for Breakthough.
#[derive(Default, Copy, Clone)]
pub struct BreakthroughBuilder {
    /// Board size.
    pub size: usize,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
#[async_trait]
impl GameBuilder for BreakthroughBuilder {
    type G = Breakthrough;

    async fn create(&self, turn: Color) -> Breakthrough {
        let mut rng = rand::thread_rng();
        let K = self.size;
        let mut content = Array::from_elem([K, K], Cell::Empty);
        let mut transposition = Array::from_elem([2, K, K], 0);

        for (x, mut column) in content.axis_iter_mut(Axis(0)).enumerate() {
            column[0] = Cell::C(Color::Black);
            column[1] = Cell::C(Color::Black);
            column[K - 2] = Cell::C(Color::White);
            column[K - 1] = Cell::C(Color::White);

            for y in 0..K {
                transposition[[Color::Black as usize, x, y]] = rng.gen::<usize>();
                transposition[[Color::White as usize, x, y]] = rng.gen::<usize>()
            }
        }

        Breakthrough {
            turn,
            content,
            transposition,
            hash: 0,
            K,
        }
    }
}

impl SingleWinner for Breakthrough {
    fn winner(&self) -> Option<Self::Player> {
        let mut some_black = false;
        let mut some_white = false;

        for i in 0..self.K {
            if self.content[[i, self.K - 1]] == Cell::C(Color::Black) {
                return Some(Color::Black);
            } else if self.content[[i, 0]] == Cell::C(Color::White) {
                return Some(Color::White);
            }

            for j in 0..self.K {
                if self.content[[i, j]] == Cell::C(Color::White) {
                    some_white = true;
                } else if self.content[[i, j]] == Cell::C(Color::Black) {
                    some_black = true;
                }
            }
        }

        if !some_white {
            Some(Color::Black)
        } else if !some_black {
            Some(Color::White)
        } else {
            None
        }
    }
}

impl Game for Breakthrough {
    type Player = Color;
    fn players() -> Vec<Color> {
        vec![Color::Black, Color::White]
    }

    fn player_after(player: Self::Player) -> Self::Player {
        player.adv()
    }

    fn turn(&self) -> Color {
        self.turn
    }
}

impl Base for Breakthrough {
    type Move = Move;

    fn possible_moves(&self) -> Vec<Self::Move> {
        if self.is_finished() {
            return vec![];
        }
        let mut res = vec![];
        for x in 0..self.K {
            for y in 0..self.K {
                if self.content[[x, y]] == Cell::C(self.turn) {
                    for direction in &[
                        MoveDirection::Front,
                        MoveDirection::FrontLeft,
                        MoveDirection::FrontRight,
                    ] {
                        let action = Move {
                            color: self.turn,
                            x,
                            y,
                            direction: *direction,
                        };
                        if action.is_valid(self.content.view()).is_some() {
                            res.push(action)
                        }
                    }
                }
            }
        }
        res
    }

    fn is_finished(&self) -> bool {
        self.winner().is_some()
    }
}

#[async_trait]
impl Playable for Breakthrough {
    async fn play(&mut self, m: &Move) -> f32 {
        if m.color != self.turn() {
            panic!("Wait. Not your turn. {:?}\n => {:?}", self, m);
        }
        match m.is_valid(self.content.view()) {
            None => -1.,
            Some((px, py)) => {
                let mut c_hash = 0;
                if let Cell::C(color) = self.content[[m.x, m.y]] {
                    // remove cell from initial position
                    c_hash ^= self.transposition[[color as usize, m.x, m.y]];
                    // add cell to new position
                    c_hash ^= self.transposition[[color as usize, px, py]];
                }
                if let Cell::C(color) = self.content[[px, py]] {
                    // eat the other cell
                    c_hash ^= self.transposition[[color as usize, px, py]];
                }
                self.hash ^= c_hash;
                assert_eq!(self.content[[m.x, m.y]], Cell::C(self.turn));
                assert_ne!(self.content[[px, py]], Cell::C(self.turn));

                self.content[[px, py]] = self.content[[m.x, m.y]];
                self.content[[m.x, m.y]] = Cell::Empty;

                let reward = if self.winner() == Some(self.turn()) {
                    1.
                } else {
                    0.
                };
                self.turn = self.turn.adv();
                reward
            }
        }
    }
}

impl Breakthrough {
    /// Prints a nice representation of the board.
    pub fn show(&self) {
        println!("{:?}", self);
    }
}

impl Features for Breakthrough {
    type StateDim = ndarray::Ix3;
    type ActionDim = ndarray::Ix3;

    type Descriptor = usize;

    fn get_features(&self) -> Self::Descriptor {
        self.K
    }

    fn state_dimension(K: &Self::Descriptor) -> Self::StateDim {
        ndarray::Dim([*K, *K, 3])
    }

    fn action_dimension(K: &Self::Descriptor) -> Self::ActionDim {
        ndarray::Dim([*K, *K, 3])
    }

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim> {
        let ft = self.get_features();
        let mut features = ndarray::Array::zeros(Self::state_dimension(&ft));

        for ((x, y, z), row) in features.indexed_iter_mut() {
            if (z == 0 && self.content[[x, y]] == Cell::C(pov))
                || (z == 1 && self.content[[x, y]] == Cell::C(pov.adv()))
            {
                *row = 1.0
            } else if z == 2 {
                if self.turn() == Color::White {
                    *row = 1.0
                } else {
                    *row = -1.0
                }
            }
        }

        features
    }

    fn moves_to_feature(
        descr: &Self::Descriptor,
        moves: &HashMap<Self::Move, f32>,
    ) -> Array<f32, Self::ActionDim> {
        let mut features = ndarray::Array::zeros(Self::action_dimension(descr));

        for (action, proba) in moves.iter() {
            features[[action.x, action.y, action.direction as usize]] = *proba;
        }

        features
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        let z: f32 = self
            .possible_moves()
            .iter()
            .map(|m| features[[m.x, m.y, m.direction as usize]])
            .sum();
        HashMap::from_iter(
            self.possible_moves()
                .iter()
                .map(|m| (*m, features[[m.x, m.y, m.direction as usize]] / z)),
        )
    }

    fn all_feature_to_moves(
        descr: &Self::Descriptor,
        features: &Array<f32, Self::ActionDim>,
    ) -> HashMap<Self::Move, f32> {
        let possible_moves = Self::all_possible_moves(descr);

        let z: f32 = possible_moves
            .iter()
            .map(|m| features[[m.x, m.y, m.direction as usize]])
            .sum();
        HashMap::from_iter(
            possible_moves
                .iter()
                .map(|m| (*m, features[[m.x, m.y, m.direction as usize]] / z)),
        )
    }

    fn all_possible_moves(K: &Self::Descriptor) -> Vec<Self::Move> {
        let mut res = vec![];
        for x in 0..*K {
            for y in 0..*K {
                for color in &[Color::Black, Color::White] {
                    for direction in &[
                        MoveDirection::Front,
                        MoveDirection::FrontLeft,
                        MoveDirection::FrontRight,
                    ] {
                        res.push(Move {
                            x,
                            y,
                            color: *color,
                            direction: *direction,
                        })
                    }
                }
            }
        }
        res
    }
}
