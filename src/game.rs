use ansi_term::Colour::{Black, White};
use ansi_term::Style;
use rand::seq::SliceRandom;
use std::fmt;

use unwrap::*;

const K: usize = 5;

#[derive(Clone, Copy, PartialEq)]
pub enum Color {
    Black,
    White,
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

#[derive(Clone, Copy, PartialEq)]
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
        let delta_x = if self.direction == MoveDirection::Front {
            0
        } else if self.direction == MoveDirection::FrontLeft {
            delta_y
        } else {
            -delta_y
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

#[derive(Clone, Copy)]
pub struct Board {
    content: [[Cell; K]; K],
}

impl Board {
    pub fn new() -> Board {
        let mut content = [[Cell::Empty; K]; K];
        for x in 0..K {
            content[x][0] = Cell::C(Color::Black);
            content[x][1] = Cell::C(Color::Black);
            content[x][K - 2] = Cell::C(Color::White);
            content[x][K - 1] = Cell::C(Color::White);
        }

        Board { content }
    }

    pub fn play(&mut self, m: &Move) {
        match m.is_valid(&self) {
            None => panic!("Wait. This is illegal. "),
            Some((px, py)) => {
                self.content[px][py] = self.content[m.x][m.y];
                self.content[m.x][m.y] = Cell::Empty;
            }
        }
    }

    pub fn show(&self) {
        println!("╔{}══╗", "══╤".repeat(K - 1));
        for y in 0..K {
            if y != 0 {
                println!("╟{}──╢", "──┼".repeat(K - 1));
            }
            print!("║");
            for x in 0..K {
                if x == 0 {
                    print!("{:?}", self.content[x][y]);
                } else {
                    print!("│{:?}", self.content[x][y]);
                }
            }
            println!("║");
        }
        println!("╚{}══╝", "══╧".repeat(K - 1));
    }

    pub fn possible_moves(&self, c: Color) -> Vec<Move> {
        let mut result = vec![];

        for y in 0..K {
            for x in 0..K {
                if self.content[x][y] == Cell::C(c) {
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

    pub fn playout(&self, c: Color, debug: bool) -> Color {
        let mut b: Board = *self;
        let mut turn = c;
        while {
            let actions = b.possible_moves(turn);
            match actions.choose(&mut rand::thread_rng()) {
                None => {
                    turn = turn.adv();
                    true
                }
                Some(action) => {
                    b.play(&action);
                    if debug {
                        println!("Turn: {:?}", turn);
                        b.show();
                    }
                    match b.winner() {
                        None => {
                            turn = turn.adv();
                            true
                        }
                        _ => false,
                    }
                }
            }
        } {}
        unwrap!(b.winner())
    }
}
