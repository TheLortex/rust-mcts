#![allow(dead_code)]
#![allow(non_snake_case)]

use indicatif::ProgressBar;
use rayon::prelude::*;

pub mod game;
pub mod policies;

use self::game::{Board, Color};
use self::policies::*;

fn monte_carlo_duel<T1: Policy, T2: Policy>(start: Color) -> Color {
    let mut b = Board::new(start);

    let mut p1 = T1::new(Color::White);
    let mut p2 = T2::new(Color::Black);

    while {
        let action = if b.turn() == Color::White {
            p1.play(&b)
        } else {
            p2.play(&b)
        };
        //b.show();
        if let Some(action) = action {
            b.play(&action);
        } else {
            b.pass();
        }
        b.winner() == None
    } {}
    b.winner().unwrap()
}

fn main() {
    let n = 100;
    let bar = ProgressBar::new(n);
    let count_victory: usize = (0..n)
        .into_par_iter()
        .map(|_| {
            bar.inc(1);
            if monte_carlo_duel::<FlatUCBMonteCarloPolicy, UCTPolicy>(Color::random())
                == Color::Black
            {
                1
            } else {
                0
            }
        })
        .sum();

    bar.finish();
    println!("Result: {}", count_victory);
}
