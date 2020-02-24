use ansi_term::Colour::Fixed;
use ansi_term::Style;
use rand::Rng;
use std::fmt;

use std::hash::*;

use super::{Game, InteractiveGame};

const K: usize = 8;

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
    color: Color,
    x: usize,
    y: usize,
    direction: MoveDirection,
}

impl Move {
    fn target(&self, b: &Breakthrough) -> (usize, usize) {
        let c = b.content[self.x][self.y];
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

    fn is_valid(&self, b: &Breakthrough) -> Option<(usize, usize)> {
        let c = b.content[self.x][self.y];
        if c != Cell::C(self.color) {
            return None;
        }
        let (px, py) = self.target(b);
        if px < K && py < K {
            if self.direction == MoveDirection::Front {
                if b.content[px][py] == Cell::Empty {
                    Some((px, py))
                } else {
                    None
                }
            } else if b.content[px][py] == Cell::Empty || b.content[px][py] != c {
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Breakthrough {
    content: [[Cell; K]; K],
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

impl Game for Breakthrough {
    type Player = Color;
    type Move = Move;

    type GameHash = usize;

    fn new(turn: Color) -> Breakthrough {
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

        Breakthrough {
            turn,
            content,
            transposition,
            hash: 0,
        }
    }

    fn players() -> Vec<Color> {
        vec![Color::Black, Color::White]
    }

    fn score(&self, c: Self::Player) -> f64 {
        match self.winner() {
            Some(c_) if c == c_ => 1.,
            Some(_) => -1.,
            _ => 0.,
        }
    }

    fn play(&mut self, m: &Move) {
        if m.color != self.turn() {
            panic!("Wait. Not your turn.");
        }
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

    fn turn(&self) -> Color {
        self.turn
    }

    fn hash(&self) -> usize {
        (self.hash << 1) + (self.turn as usize)
    }

    fn possible_moves(&self) -> Vec<Move> {
        self.possible_moves_for(self.turn)
    }

    fn winner(&self) -> Option<Color> {
        for i in 0..K {
            if self.content[i][K - 1] == Cell::C(Color::Black) {
                return Some(Color::Black);
            } else if self.content[i][0] == Cell::C(Color::White) {
                return Some(Color::White);
            }
        }
        if self.possible_moves_for(Color::Black).is_empty() {
            Some(Color::White)
        } else if self.possible_moves_for(Color::White).is_empty() {
            Some(Color::Black)
        } else {
            None
        }
    }

    fn pass(&mut self) {
        self.turn = self.turn.adv()
    }
}

impl Breakthrough {
    pub fn show(&self) {
        println!("{:?}", self);
    }

    fn possible_moves_for(&self, color: Color) -> Vec<Move> {
        let mut result = vec![];

        for y in 0..K {
            for x in 0..K {
                if self.content[x][y] == Cell::C(color) {
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
                        match m.is_valid(self) {
                            None => (),
                            Some(_) => result.push(m),
                        }
                    }
                }
            }
        }

        result
    }
}

pub struct IBreakthrough {
    game: Breakthrough,
    choosing_move: Option<PendingMove>,
    choosing_move_cb: Option<Box<dyn FnOnce(Move, &mut IBreakthrough)>>,
}

use cursive::direction::Direction;
use cursive::event::{Event, EventResult, Key, MouseButton, MouseEvent};
use cursive::theme;
use cursive::theme::ColorStyle;
use cursive::views::{Button, Dialog, LinearLayout, Panel, SelectView};
use cursive::Cursive;
use cursive::Printer;
use cursive::Vec2;

impl IBreakthrough {
    fn handle_move(&mut self, dx: isize, dy: isize) -> EventResult {
        if let Some(m) = &self.choosing_move {
            match m {
                PendingMove::SelectingPosition(x, y) => {
                    let x = *x;
                    let y = *y;
                    let new_m = self
                        .game
                        .possible_moves()
                        .iter()
                        .filter(|m| {
                            (dx != 0 && (m.x as isize - x as isize) * dx > 0)
                                || (dy != 0 && (m.y as isize - y as isize) * dy > 0)
                        })
                        .min_by_key(|m| {
                            let a = m.x as isize - x as isize;
                            let b = m.y as isize - y as isize;
                            if dx == 0 {
                                (b * dy, a.abs())
                            } else {
                                (a * dx, b.abs())
                            }
                        })
                        .cloned();
                    if let Some(new_m) = new_m {
                        self.choosing_move = Some(PendingMove::SelectingPosition(new_m.x, new_m.y))
                    }
                }
                PendingMove::SelectingMove(m) => {
                    if dx == 0 {
                        self.choosing_move = Some(PendingMove::SelectingPosition(m.x, m.y))
                    } else {
                        let new_m = MoveDirection::all()
                            .iter()
                            .map(|d| Move {
                                color: self.game.turn(),
                                direction: *d,
                                x: m.x,
                                y: m.y,
                            })
                            .filter(|m| m.is_valid(&self.game).is_some())
                            .filter(|m2| {
                                let m2_t = m2.target(&self.game);
                                let m_t = m.target(&self.game);
                                (m2_t.0 as isize - m_t.0 as isize) * dx > 0
                            })
                            .min_by_key(|m2| {
                                let m2_t = m2.target(&self.game);
                                let m_t = m.target(&self.game);
                                (m2_t.0 as isize - m_t.0 as isize) * dx
                            });
                        if let Some(new_m) = new_m {
                            self.choosing_move = Some(PendingMove::SelectingMove(new_m))
                        }
                    }
                }
            };
            EventResult::Consumed(None)
        } else {
            EventResult::Ignored
        }
    }
}

impl cursive::view::View for IBreakthrough {
    fn draw(&self, printer: &Printer) {
        let black_color = ColorStyle::new(
            theme::Color::RgbLowRes(0, 0, 0),
            theme::Color::TerminalDefault,
        );

        let white_color = ColorStyle::new(
            theme::Color::RgbLowRes(5, 3, 5),
            theme::Color::TerminalDefault,
        );

        printer.print((1, 1), &format!("╔{}══╗", "══╤".repeat(K - 1)));
        for y in 0..K {
            if y != 0 {
                printer.print((1, 1 + 2 * y), &format!("╟{}──╢", "──┼".repeat(K - 1)));
            }
            printer.print((1, 2 + 2 * y), "║");
            for x in 0..K {
                if x != 0 {
                    printer.print((1 + 3 * x, 2 + 2 * y), "│")
                };

                match self.game.content[x][y] {
                    Cell::Empty => (),
                    Cell::C(Color::Black) => printer.with_color(black_color, |printer| {
                        printer.print((2 + 3 * x, 2 + 2 * y), "▓▓")
                    }),
                    Cell::C(Color::White) => printer.with_color(white_color, |printer| {
                        printer.print((2 + 3 * x, 2 + 2 * y), "▓▓")
                    }),
                }
            }
            printer.print((1 + 3 * K, 2 + 2 * y), "║");
        }
        printer.print((1, 1 + 2 * K), &format!("╚{}══╝", "══╧".repeat(K - 1)));

        let select_color = ColorStyle::new(
            theme::Color::RgbLowRes(1, 1, 1),
            theme::Color::RgbLowRes(4, 4, 4),
        );

        if let Some(m) = self.choosing_move {
            let (x, y) = match m {
                PendingMove::SelectingPosition(x, y) => (x, y),
                PendingMove::SelectingMove(m) => (m.x, m.y),
            };
            printer.with_color(select_color, |printer| {
                printer.print((1 + 3 * x, 1 + 2 * y), "┏━━┓");
                printer.print((1 + 3 * x, 2 + 2 * y), "┣");
                printer.print((4 + 3 * x, 2 + 2 * y), "┫");
                printer.print((1 + 3 * x, 3 + 2 * y), "┗━━┛");
            });

            if let PendingMove::SelectingMove(mv) = m {
                for direction in &[
                    MoveDirection::Front,
                    MoveDirection::FrontLeft,
                    MoveDirection::FrontRight,
                ] {
                    let m = Move {
                        color: self.game.turn(),
                        x,
                        y,
                        direction: *direction,
                    };
                    match m.is_valid(&self.game) {
                        None => (),
                        Some((px, py)) => {
                            let (px, py, color) = if *direction == mv.direction {
                                (
                                    px,
                                    py,
                                    ColorStyle::new(
                                        theme::Color::RgbLowRes(5, 0, 0),
                                        theme::Color::RgbLowRes(4, 4, 4),
                                    ),
                                )
                            } else {
                                (
                                    px,
                                    py,
                                    ColorStyle::new(
                                        theme::Color::RgbLowRes(0, 0, 0),
                                        theme::Color::RgbLowRes(4, 4, 4),
                                    ),
                                )
                            };
                            printer.with_color(color, |printer| {
                                printer.print((1 + 3 * px, 1 + 2 * py), "┼──┼");
                                printer.print((1 + 3 * px, 2 + 2 * py), "│▒▒│");
                                printer.print((1 + 3 * px, 3 + 2 * py), "┼──┼");
                            });
                        }
                    };
                }
            }
        }
    }

    fn take_focus(&mut self, _: Direction) -> bool {
        true
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        match event {
            Event::Key(Key::Right) => self.handle_move(1, 0),
            Event::Key(Key::Left) => self.handle_move(-1, 0),
            Event::Key(Key::Up) => self.handle_move(0, -1),
            Event::Key(Key::Down) => self.handle_move(0, 1),
            Event::Key(Key::Enter) => {
                if let Some(m) = self.choosing_move {
                    match m {
                        PendingMove::SelectingMove(m) => {
                            if let Some(f) = self.choosing_move_cb.take() {
                                f(m, self);
                            }
                            self.choosing_move = None;
                            EventResult::Consumed(None)
                        }
                        PendingMove::SelectingPosition(x, y) => {
                            self.choosing_move = Some(PendingMove::SelectingMove(
                                *self
                                    .game
                                    .possible_moves()
                                    .iter()
                                    .find(|m| m.x == x && m.y == y)
                                    .unwrap(),
                            ));
                            EventResult::Consumed(None)
                        }
                    }
                } else {
                    EventResult::Ignored
                }
            }
            _ => EventResult::Ignored,
        }
    }

    fn required_size(&mut self, _: Vec2) -> Vec2 {
        Vec2 {
            x: K * 3 + 3,
            y: K * 2 + 3,
        }
    }
}

impl InteractiveGame for IBreakthrough {
    type G = Breakthrough;

    fn new(turn: <Breakthrough as Game>::Player) -> Self {
        IBreakthrough {
            game: Breakthrough::new(turn),
            choosing_move: None,
            choosing_move_cb: None,
        }
    }

    fn get(&self) -> &Self::G {
        &self.game
    }

    fn get_mut(&mut self) -> &mut Self::G {
        &mut self.game
    }

    fn choose_move(&mut self, cb: Box<dyn FnOnce(<Self::G as Game>::Move, &mut Self)>) {
        let first_move = self.game.possible_moves()[0];
        self.choosing_move = Some(PendingMove::SelectingPosition(first_move.x, first_move.y));
        self.choosing_move_cb = Some(cb)
    }
}
