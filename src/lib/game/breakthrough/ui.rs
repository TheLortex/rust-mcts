use super::*;

pub struct IBreakthrough {
    game: Breakthrough,
    choosing_move: Option<PendingMove>,
    choosing_move_cb: Option<Box<dyn FnOnce(Move, &mut IBreakthrough)>>,
}

use cursive::direction::Direction;
use cursive::event::{Event, EventResult, Key};
use cursive::theme;
use cursive::theme::ColorStyle;
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
                            .filter(|m| m.is_valid(&self.game.content).is_some())
                            .filter(|m2| {
                                let m2_t = m2.target();
                                let m_t = m.target();
                                (m2_t.0 as isize - m_t.0 as isize) * dx > 0
                            })
                            .min_by_key(|m2| {
                                let m2_t = m2.target();
                                let m_t = m.target();
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
        // print letters
        for x in 0..K {
            printer.print((2+3*x, 0), &(('a' as usize +x) as u8 as char).to_string());
            printer.print((0, 2+2*x), &format!("{}", 1+x));
        }
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
                    match m.is_valid(&self.game.content) {
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


    fn new(turn: <Breakthrough as MultiplayerGame>::Player) -> Self {
        IBreakthrough {
            game: (BreakthroughBuilder {}).create(turn),
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

    fn choose_move(&mut self, cb: Box<dyn FnOnce(<Self::G as BaseGame>::Move, &mut Self)>) {
        let first_move = self.game.possible_moves()[0];
        self.choosing_move = Some(PendingMove::SelectingPosition(first_move.x, first_move.y));
        self.choosing_move_cb = Some(cb)
    }
}
