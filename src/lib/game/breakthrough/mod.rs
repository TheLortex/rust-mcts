use crate::game::{Base, Feature, Game, GameBuilder, InteractiveGame, Playable};
use crate::settings::BREAKTHROUGH_K as K;

use ansi_term::Colour::Fixed;
use ansi_term::Style;
use rand::Rng;
use std::fmt;
use std::hash::*;

pub mod ui;
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

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Move {
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

    pub fn is_valid(&self, content: &[[Cell; K]; K]) -> Option<(usize, usize)> {
        let c = content[self.x][self.y];
        if c != Cell::C(self.color) {
            return None;
        }
        let (px, py) = self.target();
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

use arrayvec::ArrayVec;
#[derive(Clone, Eq)]
pub struct Breakthrough {
    pub content: [[Cell; K]; K],

    positions_black: ArrayVec<[(usize, usize); 2 * K]>,
    positions_white: ArrayVec<[(usize, usize); 2 * K]>,

    transposition: [[[usize; K]; K]; 2],
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

#[derive(Default, Copy, Clone)]
pub struct BreakthroughBuilder {}

impl GameBuilder<Breakthrough> for BreakthroughBuilder {
    fn create(&self, turn: Color) -> Breakthrough {
        let mut rng = rand::thread_rng();

        let mut content = [[Cell::Empty; K]; K];
        let mut transposition = [[[0; K]; K]; 2];
        let mut positions_black = ArrayVec::new();
        let mut positions_white = ArrayVec::new();

        for (x, column) in content.iter_mut().enumerate() {
            column[0] = Cell::C(Color::Black);
            column[1] = Cell::C(Color::Black);
            column[K - 2] = Cell::C(Color::White);
            column[K - 1] = Cell::C(Color::White);

            positions_black.push((x, 0));
            positions_black.push((x, 1));
            positions_white.push((x, K - 1));
            positions_white.push((x, K - 2));

            for y in 0..K {
                transposition[Color::Black as usize][x][y] = rng.gen::<usize>();
                transposition[Color::White as usize][x][y] = rng.gen::<usize>()
            }
        }

        Breakthrough {
            turn,
            content,
            positions_black,
            positions_white,
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
        if self.positions_black.is_empty() {
            Some(Color::White)
        } else if self.positions_white.is_empty() {
            Some(Color::Black)
        } else {
            None
        }
    }

    fn remove_player(&mut self, x: usize, y: usize, color: Color) {
        let target = if color == Color::Black {
            &mut self.positions_black
        } else {
            &mut self.positions_white
        };

        {
            for (i, (px, py)) in target.iter().enumerate() {
                if x == *px && y == *py {
                    target.swap_remove(i);
                    break;
                }
            }
        }
    }

    fn add_player(&mut self, x: usize, y: usize, color: Color) {
        let target = if color == Color::Black {
            &mut self.positions_black
        } else {
            &mut self.positions_white
        };

        target.push((x, y));
    }
}

impl Game for Breakthrough {
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

type PossibleMovesIterator<'a> = impl Iterator<Item = Move> + 'a;

impl Base for Breakthrough {
    type Move = Move;
    type MoveIterator<'a> = PossibleMovesIterator<'a>;
    /*
        fn hash(&self) -> usize {
            (self.hash << 1) + (self.turn as usize)
        }
    */
    fn possible_moves<'a>(&'a self) -> Self::MoveIterator<'a> {
        let target: &'a ArrayVec<_> = if self.turn == Color::Black {
            &self.positions_black
        } else {
            &self.positions_white
        };

        let color = self.turn;
        let content = self.content;

        target.iter().flat_map(move |(x, y)| {
            [
                MoveDirection::Front,
                MoveDirection::FrontLeft,
                MoveDirection::FrontRight,
            ]
            .iter()
            .map(move |direction| Move {
                color,
                x: *x,
                y: *y,
                direction: *direction,
            })
            .filter(move |action| action.is_valid(&content).is_some())
        })
    }

    fn is_finished(&self) -> bool {
        self.winner().is_some()
    }
}

impl Playable for Breakthrough {
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

                if let Cell::C(_) = self.content[px][py] {
                    self.remove_player(px, py, self.turn.adv());
                };
                self.remove_player(m.x, m.y, self.turn);
                self.add_player(px, py, self.turn);

                self.content[px][py] = self.content[m.x][m.y];
                self.content[m.x][m.y] = Cell::Empty;

                self.turn = self.turn.adv();
            }
        }
    }
}

impl Breakthrough {
    pub fn show(&self) {
        println!("{:?}", self);
    }
}

use ndarray::Array;
use std::collections::HashMap;
use std::iter::FromIterator;

impl Feature for Breakthrough {
    type StateDim = ndarray::Ix3;
    type ActionDim = ndarray::Ix3;

    fn state_dimension() -> Self::StateDim {
        ndarray::Dim([K, K, 3])
    }

    fn action_dimension() -> Self::ActionDim {
        ndarray::Dim([K, K, 3])
    }

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim> {
        let mut features = ndarray::Array::zeros(Self::state_dimension());

        for ((x, y, z), row) in features.indexed_iter_mut() {
            if (z == 0 && self.content[x][y] == Cell::C(pov))
                || (z == 1 && self.content[x][y] == Cell::C(pov.adv()))
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

    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim> {
        let mut features = ndarray::Array::zeros(Self::action_dimension());

        for (action, proba) in moves.iter() {
            features[[action.x, action.y, action.direction as usize]] = *proba;
        }

        features
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        let z: f32 = self
            .possible_moves()
            .map(|m| features[[m.x, m.y, m.direction as usize]])
            .sum();
        HashMap::from_iter(
            self.possible_moves()
                .map(|m| (m, features[[m.x, m.y, m.direction as usize]] / z)),
        )
    }
}
#[allow(clippy::float_cmp)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_to_moves() {
        let g: Breakthrough = (BreakthroughBuilder {}).create(Color::Black);
        let mut features = Array::zeros(Breakthrough::action_dimension());
        features[[0, 1, MoveDirection::Front as usize]] = 0.2;
        features[[1, 1, MoveDirection::Front as usize]] = 0.2;
        features[[0, 0, MoveDirection::Front as usize]] = 0.6;

        let moves = g.feature_to_moves(&features);
        assert_eq!(
            *moves
                .get(&Move {
                    x: 0,
                    y: 1,
                    color: Color::Black,
                    direction: MoveDirection::Front
                })
                .unwrap(),
            0.5
        );
        assert_eq!(
            *moves
                .get(&Move {
                    x: 1,
                    y: 1,
                    color: Color::Black,
                    direction: MoveDirection::Front
                })
                .unwrap(),
            0.5
        );
    }

    #[test]
    fn test_initial_state_to_features() {
        for color in &[Color::Black, Color::White] {
            for pov in &[Color::Black, Color::White] {
                let g: Breakthrough = (BreakthroughBuilder {}).create(*color);
                let f = g.state_to_feature(*pov);

                if *pov == Color::Black {
                    assert_eq!(f[[2, 0, 0]], 1.0);
                    assert_eq!(f[[2, K - 1, 1]], 1.0);
                } else {
                    assert_eq!(f[[2, K - 1, 0]], 1.0);
                    assert_eq!(f[[2, 0, 1]], 1.0);
                }

                if *color == Color::Black {
                    assert_eq!(f[[0, 2, 2]], -1.0);
                } else {
                    assert_eq!(f[[0, 2, 2]], 1.0);
                }
            }
        }
    }
}
