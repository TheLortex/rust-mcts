#![allow(dead_code)]
#![allow(non_snake_case)]
#![feature(associated_type_defaults)]
#![feature(test)]
#![allow(unused_imports)]
#![feature(fn_traits)]

use cursive::direction::Direction;
use cursive::event::{Event, EventResult, MouseButton, MouseEvent};
use cursive::theme::{BaseColor, Color, ColorStyle};
use cursive::views::{Button, Dialog, LinearLayout, NamedView, Panel, SelectView};
use cursive::Cursive;
use cursive::Printer;
use cursive::Vec2;

use std::cell::RefCell;
use std::rc::Rc;

use indicatif::ProgressBar;
use rayon::prelude::*;

pub mod game;
pub mod mcts;
pub mod nmcs;
pub mod policies;

#[cfg(test)]
mod tests;

use self::game::breakthrough::*;
use self::game::misere_breakthrough::*;
use self::game::{Game, InteractiveGame};
use self::mcts::*;
use self::nmcs::*;
use self::policies::*;

fn game_duel<G: Game, P1: Policy<G>, P2: Policy<G>>(
    start: G::Player,
    p1: &mut P1,
    p2: &mut P2,
) -> G::Player {
    let mut b = G::new(start);
    const DBG: bool = false;

    while {
        let action = if b.turn() == G::players()[0] {
            p1.play(&b)
        } else {
            p2.play(&b)
        };
        if DBG {
            println!("{:?} => {:?}", b, action);
        }
        if let Some(action) = action {
            b.play(&action);
        } else {
            b.pass();
        }
        b.winner() == None
    } {}
    if DBG {
        println!("{:?}", b);
    };
    b.winner().unwrap()
}

pub fn monte_carlo_match<G: Game, P1: PolicyBuilder<G> + Sync, P2: PolicyBuilder<G> + Sync>(
    n: usize,
    pb1: &P1,
    pb2: &P2,
) -> usize {
    let bar = ProgressBar::new(n as u64);

    let count_victory: usize = (0..n)
        .into_par_iter()
        .map(|_| {
            let mut p1 = pb1.create(G::players()[0]);
            let mut p2 = pb2.create(G::players()[1]);

            let result = if game_duel(G::players()[0], &mut p1, &mut p2) == G::players()[1] {
                1
            } else {
                0
            };
            bar.inc(1);
            result
        })
        .sum();
    bar.finish();
    count_victory
}

struct GameDuelUI {}

impl GameDuelUI {
    fn render<IG: InteractiveGame, P1: Policy<IG::G> + 'static, P2: Policy<IG::G> + 'static>(
        start: <IG::G as Game>::Player,
        p1: P1,
        p2: P2,
    ) -> impl cursive::view::View {
        let r_p1 = RefCell::new(p1);
        let r_p2 = RefCell::new(p2);

        LinearLayout::vertical()
            .child(NamedView::new("game", IG::new(start)))
            .child(Button::new_raw("Next", move |mut s| {
                let state: &mut IG = &mut s.find_name("game").unwrap();

                let mut p1 = r_p1.borrow_mut();
                let mut p2 = r_p2.borrow_mut();
                let p1_to_play = state.get().turn() == <IG::G as Game>::players()[0];

                if p1_to_play {
                    //p1.play(&state)
                    state.choose_move(Box::new(|action, state| state.get_mut().play(&action)))
                } else {
                    let action = p2.play(state.get()).unwrap();
                    state.get_mut().play(&action);
                };
            }))
    }
}

type G = Breakthrough;

fn main() {
    /*let p1 = UCT::default();
    let p2 = RAVE::default();

    println!("Result: {}", monte_carlo_match::<G, _, _>(1, &p1, &p2));*/

    let mut siv = Cursive::default();

    let pb1 = Random::default();
    let p1: RandomPolicy<G> = pb1.create(G::players()[0]);
    let pb2 = UCT::default();
    let p2: UCTPolicy<G> = pb2.create(G::players()[1]);

    siv.add_layer(
        Dialog::new()
            .title("Breakthrough")
            .content(GameDuelUI::render::<IBreakthrough, _, _>(
                G::players()[0],
                p1,
                p2,
            )),
    );

    siv.run();
}

/* Player policy */
/*
pub struct PlayerPolicy {}

impl<G: Game> Policy<G> for PlayerPolicy {
    fn play(self: &mut PlayerPolicy, board: &G) -> Option<G::Move> {
        let moveset = board.possible_moves();

    }
}*/
