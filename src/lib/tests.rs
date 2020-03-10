
extern crate test;
use test::Bencher;

use rand::seq::SliceRandom;
use rand::Rng;

use super::*;
use super::game::*;
use super::policies::*;

type G = breakthrough::Breakthrough;

#[bench]
#[ignore]
fn bench_game_winner(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    b.iter(|| game.winner());
}

#[bench]
fn bench_game_possible_moves(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    b.iter(|| game.possible_moves());
}

#[bench]
fn bench_random_playout(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    b.iter(|| game.playout());
}

#[bench]
#[ignore]
fn bench_game_clone(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    b.iter(|| game.clone());
}


#[bench]
fn bench_random_move(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    b.iter(|| {
        let mut g = game.clone();
        g.random_move();
    });
}

#[bench]
fn bench_play(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    let actions = game.possible_moves();
    let chosen_action = actions.choose(&mut rand::thread_rng()).copied().unwrap();

    b.iter(|| {
        let mut g = game.clone();
        g.play(&chosen_action)
    });
}

#[bench]
fn bench_is_valid(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    let actions = game.possible_moves();
    let chosen_action = actions.choose(&mut rand::thread_rng()).copied().unwrap();

    b.iter(|| {
        chosen_action.is_valid(&game.content)
    });
}

#[bench]
fn bench_ppa_simulate(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    let pb = policies::ppa::PPA::<_, NoFeatures>::default();

    b.iter(|| {
        let mut policy = pb.create(G::players()[0]);
        policy.simulate(&game)
    });
}

#[bench]
fn bench_ppa_next_move(b: &mut Bencher) {
    let game = G::new(G::players()[0], ());
    let pb = policies::ppa::PPA::<_, NoFeatures>::default();

    b.iter(|| {
        let mut policy = pb.create(G::players()[0]);
        policy.next_move(&game)
    });
}


/*
#[bench]
fn bench_ppa_policy(b: &mut Bencher) {
    let game = G::new(G::players()[0]);
    let pb = PPA::<_, BTNoFeatures>::default();

    b.iter(|| {
        let mut policy = pb.create(G::players()[0]);
        policy.play(&game)
    })
}*/