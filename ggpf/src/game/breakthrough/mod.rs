use crate::game::*;
use crate::settings::BREAKTHROUGH_K as K;

use ansi_term::Colour::Fixed;
use ansi_term::Style;
use async_trait::async_trait;
use ndarray::Array;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;
use std::hash::*;
use std::iter::FromIterator;

/**
 *  Breakthrough interactive interface.
 */
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

use arrayvec::ArrayVec;
/// Breakthrough game state instance
#[derive(Clone, Eq)]
pub struct Breakthrough {
    content: [[Cell; K]; K],

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

/// Game builder for Breakthough.
///
/// This game builder has no settings.
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

impl SingleWinner for Breakthrough {
    fn winner(&self) -> Option<Self::Player> {
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
}

impl Breakthrough {
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
        let target: &ArrayVec<_> = if self.turn == Color::Black {
            &self.positions_black
        } else {
            &self.positions_white
        };

        let color = self.turn;
        let content = self.content;

        if self.is_finished() {
            vec![]
        } else {
            target
                .iter()
                .flat_map(move |(x, y)| {
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
                .collect()
        }
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
        match m.is_valid(&self.content) {
            None => -1.,
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

    type Descriptor = ();

    fn get_features(&self) -> Self::Descriptor {
        ()
    }

    fn state_dimension(descr: &Self::Descriptor) -> Self::StateDim {
        ndarray::Dim([K, K, 3])
    }

    fn action_dimension(descr: &Self::Descriptor) -> Self::ActionDim {
        ndarray::Dim([K, K, 3])
    }

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim> {
        let ft = self.get_features();
        let mut features = ndarray::Array::zeros(Self::state_dimension(&ft));

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

    fn all_possible_moves(descr: &Self::Descriptor) -> Vec<Self::Move> {
        let mut res = vec![];
        for x in 0..K {
            for y in 0..K {
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
#[allow(clippy::float_cmp)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_to_moves() {
        let g: Breakthrough = (BreakthroughBuilder {}).create(Color::Black);
        let ft = g.get_features();
        let mut features = Array::zeros(Breakthrough::action_dimension(&ft));
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