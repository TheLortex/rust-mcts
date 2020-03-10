use rand::seq::SliceRandom;
use std::fmt::Debug;

use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

pub mod breakthrough;
pub mod misere_breakthrough;
pub mod weak_schur;
pub mod hashcode_20;

/* A MULTI-PLAYER GAME */
pub trait Game: Sized + Clone + Debug {
    type Player: PartialEq + Eq + Copy + Clone + Debug;
    type Move: PartialEq + Eq + Copy + Clone + Hash + Debug;
    type GameHash: PartialEq + Eq + Copy + Clone + Hash + Debug;
    type Settings: Sync + Clone + Debug;

    fn new(turn: Self::Player, settings: Self::Settings) -> Self;

    fn players() -> Vec<Self::Player>;
    fn hash(&self) -> Self::GameHash;
    fn turn(&self) -> Self::Player;
    fn possible_moves(&self) -> &Vec<Self::Move>;

    fn play(&mut self, action: &Self::Move);
    fn pass(&mut self);
    fn winner(&self) -> Option<Self::Player>;
    fn score(&self, player: Self::Player) -> f32;

    fn random_move(&mut self) -> (Self::GameHash, Option<Self::Move>) {
        let actions = self.possible_moves();
        let chosen_action = actions.choose(&mut rand::thread_rng()).copied();

        let gh = self.hash();

        match chosen_action {
            None => self.pass(),
            Some(action) => self.play(&action),
        };
        (gh, chosen_action)
    }

    fn playout_full_history(&self) -> (Self, Vec<(Self, Self::Move)>) {
        let mut s = self.clone();
        let mut hist = Vec::new();

        while { s.winner().is_none() } {
            let s_cloned = s.clone();
            let (_,m) = s.random_move();
            hist.push((s_cloned, m.unwrap()));
        }
        (s, hist)
    }

    fn playout_board_history(&self) -> (Self, Vec<(Self::GameHash, Self::Move)>) {
        let mut s = self.clone();
        let mut hist = Vec::new();

        while { s.winner().is_none() } {
            let (h,m) = s.random_move();
            hist.push((h, m.unwrap()));
        }
        (s, hist)
    }

    fn playout_history(&self) -> (Self::Player, Vec<(Self::GameHash, Self::Move)>) {
        let (s, hist) = self.playout_board_history();
        (s.winner().unwrap(), hist)
    }

    fn playout_board(&self) -> Self {
        let (s, _) = self.playout_board_history();
        s
    }

    fn playout(&self) -> Self::Player {
        let (s, _) = self.playout_board_history();
        s.winner().unwrap()
    }
}
/* TODO: MoveCode */
pub trait MoveCode<G: Game> {
    fn code(game: &G, action: &G::Move) -> usize;
}

pub struct NoFeatures {}
impl<T: Game> MoveCode<T> for NoFeatures {
    fn code(_: &T, action: &T::Move) -> usize {
        let mut s = DefaultHasher::new();
        action.hash(&mut s);
        usize::try_from(s.finish()).unwrap()
    }
}

/* GAME WITH AN UI */
pub trait InteractiveGame: cursive::view::View {
    type G: Game;

    fn new(turn: <Self::G as Game>::Player) -> Self;

    fn get_mut(&mut self) -> &mut Self::G;
    fn get(&self) -> &Self::G;
    fn choose_move(&mut self, cb: Box<dyn FnOnce(<Self::G as Game>::Move, &mut Self)>);
}
