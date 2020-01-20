use rand::seq::SliceRandom;

use std::hash::*;

pub mod breakthrough;

pub trait Game: Sized + Copy + Clone {
    type Player: PartialEq + Eq + Copy + Clone;
    type Move: PartialEq + Eq + Copy + Clone + Hash;

    fn new(turn: Self::Player) -> Self;

    fn players()   -> Vec<Self::Player>;
    fn hash(&self) -> usize;
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
}
