use indicatif::ProgressBar;
use rayon::prelude::*;


pub mod policies;
pub mod game;

use self::game::{Color, Board};
use self::policies::*;

fn monte_carlo_duel<T1: Policy, T2: Policy>(start: Color) -> Color {
    let mut turn = start;
    let mut b = Board::new();

    let p1 = T1::new(Color::Black);
    let p2 = T2::new(Color::White);

    while {
        let action = if turn == Color::Black {
            p1.play(&b)
        } else {
            p2.play(&b)
        };
        if let Some(action) = action {
            b.play(&action);
        }
        turn = turn.adv();
        b.winner() == None
    } {}
    b.winner().unwrap()
}

fn main() {
    let n = 100;
    let bar = ProgressBar::new(n);
    let count_victory_ucb: usize = (0..n)
        .into_par_iter()
        .map(|_| {
            bar.inc(1);
            if monte_carlo_duel::<FlatMonteCarloPolicy,FlatUCBMonteCarloPolicy>(Color::random()) == Color::White {
                1
            } else {
                0
            }
        })
        .sum();

    bar.finish();
    println!("Result: {}", count_victory_ucb);
}
