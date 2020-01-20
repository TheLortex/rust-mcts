use ansi_term::Colour::{Black, White};
use ansi_term::Style;
use rand::seq::SliceRandom;
use std::fmt;
use rand::Rng;

use std::hash::*;
use unwrap::*;

const K: usize = 5;


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black = 0,
    White = 1,
}

impl Color {
    pub fn adv(&self) -> Color {
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
        match self {
            Color::Black => write!(f, "{}", Style::new().on(Black).paint("  ")),
            Color::White => write!(f, "{}", Style::new().on(White).paint("  ")),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    C(Color),
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cell::Empty => write!(f, "  "),
            Cell::C(c) => write!(f, "{:?}", c),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MoveDirection {
    Front,
    FrontLeft,
    FrontRight,
}

#[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Move {
    x: usize,
    y: usize,
    direction: MoveDirection,
}

impl Move {
    fn is_valid(&self, b: &Board) -> Option<(usize, usize)> {
        let c = b.content[self.x][self.y];
        if c == Cell::Empty {
            return None;
        }
        let delta_y = if c == Cell::C(Color::Black) { 1 } else { -1 };
        let delta_x = match self.direction {
            MoveDirection::Front => 0,
            MoveDirection::FrontLeft => delta_y,
            MoveDirection::FrontRight => -delta_y,
        };
        
        let px = (self.x as i32 + delta_x) as usize;
        let py = (self.y as i32 + delta_y) as usize;
        if px < K && py < K {
            if self.direction == MoveDirection::Front {
                if b.content[px][py] == Cell::Empty {
                    return Some((px, py));
                } else {
                    return None;
                }
            } else {
                if b.content[px][py] == Cell::Empty {
                    return Some((px, py));
                } else if b.content[px][py] != c {
                    return Some((px, py));
                } else {
                    return None;
                }
            }
        } else {
            return None;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Board {
    content: [[Cell; K]; K],
    transposition: [[[usize; K]; K]; 2],
    hash: usize,
    turn: Color
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Turn: {:?}\n", self.turn)?;
        write!(f, "╔{}══╗\n", "══╤".repeat(K - 1))?;
        for y in 0..K {
            if y != 0 {
                write!(f, "╟{}──╢\n", "──┼".repeat(K - 1))?;
            }
            write!(f, "║")?;
            for x in 0..K {
                if x == 0 {
                    write!(f, "{:?}", self.content[x][y])?;
                } else {
                    write!(f, "│{:?}", self.content[x][y])?;
                }
            }
            write!(f, "║\n")?;
        }
        write!(f, "╚{}══╝\n", "══╧".repeat(K - 1))
    }

}

impl Board {
    pub fn new(turn: Color) -> Board {
        let mut rng = rand::thread_rng();

        let mut content = [[Cell::Empty; K]; K];
        let mut transposition = [[[0; K]; K]; 2];
        for x in 0..K {
            content[x][0] = Cell::C(Color::Black);
            content[x][1] = Cell::C(Color::Black);
            content[x][K - 2] = Cell::C(Color::White);
            content[x][K - 1] = Cell::C(Color::White);

            for y in 0..K {
                transposition[Color::Black as usize][x][y] = rng.gen::<usize>();
                transposition[Color::White as usize][x][y] = rng.gen::<usize>()
            }
        }

        Board { turn, content, transposition, hash: 0 }
    }

    pub fn play(&mut self, m: &Move) {
        match m.is_valid(&self) {
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
                self.turn = self.turn.adv();
            }
        }
    }

    pub fn turn(&self) -> Color {
        self.turn
    }

    pub fn hash(&self) -> usize {
        2 * self.hash + (self.turn as usize)
    }

    pub fn show(&self) {
        println!("{:?}", self);
    }

    pub fn possible_moves(&self) -> Vec<Move> {
        let mut result = vec![];

        for y in 0..K {
            for x in 0..K {
                if self.content[x][y] == Cell::C(self.turn) {
                    for direction in vec![
                        MoveDirection::Front,
                        MoveDirection::FrontLeft,
                        MoveDirection::FrontRight,
                    ] {
                        let m = Move { x, y, direction };
                        match m.is_valid(self) {
                            None => (),
                            Some(_) => result.push(m),
                        }
                    }
                }
            }
        }

        return result;
    }

    pub fn winner(&self) -> Option<Color> {
        for i in 0..K {
            if self.content[i][K - 1] == Cell::C(Color::Black) {
                return Some(Color::Black);
            } else if self.content[i][0] == Cell::C(Color::White) {
                return Some(Color::White);
            }
        }
        return None;
    }

    pub fn pass(&mut self) {
        self.turn = self.turn.adv()
    }

    pub fn playout(&self, debug: bool) -> Color {
        let mut b: Board = *self;
        
        while {
            let actions = b.possible_moves();
            match actions.choose(&mut rand::thread_rng()) {
                None => {
                    b.pass();
                    true
                }
                Some(action) => {
                    b.play(&action);
                    if debug {
                        println!("Turn: {:?}", b.turn());
                        b.show();
                    }
                    match b.winner() {
                        None => true,
                        _ => false,
                    }
                }
            }
        } {}
        unwrap!(b.winner())
    }
}
