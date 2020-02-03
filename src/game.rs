use rand::seq::SliceRandom;
use std::fmt::Debug;

use std::hash::Hash;

pub mod breakthrough;


pub trait Game: Sized + Copy + Clone + Debug {
    type Player: PartialEq + Eq + Copy + Clone + Debug;
    type Move: PartialEq + Eq + Copy + Clone + Hash + Debug;
    type GameHash: PartialEq + Eq + Copy + Clone + Hash + Debug;

    fn new(turn: Self::Player) -> Self;

    fn players()   -> Vec<Self::Player>;
    fn hash(&self) -> Self::GameHash;
    fn turn(&self) -> Self::Player;
    fn possible_moves(&self) -> Vec<Self::Move>;

    fn play(&mut self, action: &Self::Move);
    fn pass(&mut self);
    fn winner(&self) -> Option<Self::Player>;

    fn playout(&self) -> Self::Player {
        let mut s = *self;

        while {
            let actions = s.possible_moves();
            match actions.choose(&mut rand::thread_rng()) {
                None => s.pass(),
                Some(action) => {
                    s.play(action)
                }
            };
            match s.winner() {
                None => true,
                Some (_) => false
            }
        } {}
        s.winner().unwrap()
    }

    fn playout_history(&self) -> (Self::Player, Vec<(Self::GameHash, Option<Self::Move>)>) {
        let mut s = *self;
        let mut hist = Vec::new();

        while {
            let actions = s.possible_moves();
            let chosen_action = actions.choose(&mut rand::thread_rng());
            hist.push((s.hash(), chosen_action.map(|x| *x)));
            match chosen_action {
                None => s.pass(),
                Some(action) => {
                    s.play(action)
                }
            };
            match s.winner() {
                None => true,
                Some (_) => false
            }
        } {}
        (s.winner().unwrap(), hist)
    }

}
