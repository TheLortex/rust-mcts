#![allow(dead_code)]
#![allow(non_snake_case)]

use indicatif::ProgressBar;
use rayon::prelude::*;

pub mod game;
pub mod policies;
pub mod mcts;

use self::game::breakthrough::*;
use self::game::Game;
use self::policies::*;
use self::mcts::RAVEPolicy;

fn monte_carlo_duel<G: Game, P1: Policy<G>, P2: Policy<G>>(start: G::Player) -> G::Player {
    let mut b = G::new(start);

    let mut p1 = P1::new(G::players()[0]);
    let mut p2 = P2::new(G::players()[1]);

    const DBG: bool = true;

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


fn main() {
    let n = 1;
    let bar = ProgressBar::new(n);
    let count_victory: usize = (0..n)
        .into_par_iter()
        .map(|_| {
            bar.inc(1);
            if monte_carlo_duel::<
                Breakthrough,
                UCTPolicy<Breakthrough>,
                RAVEPolicy<Breakthrough>,
            >(Breakthrough::players()[0]) //Color::random())
                == Breakthrough::players()[1]
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
