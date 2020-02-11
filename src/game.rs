use rand::seq::SliceRandom;
use std::fmt::Debug;

use std::hash::Hash;

pub mod breakthrough;
pub mod misere_breakthrough;

pub trait Game: Sized + Copy + Clone + Debug {
    type Player: PartialEq + Eq + Copy + Clone + Debug;
    type Move: PartialEq + Eq + Copy + Clone + Hash + Debug;
    type GameHash: PartialEq + Eq + Copy + Clone + Hash + Debug;

    fn new(turn: Self::Player) -> Self;

    fn players() -> Vec<Self::Player>;
    fn hash(&self) -> Self::GameHash;
    fn turn(&self) -> Self::Player;
    fn possible_moves(&self) -> Vec<Self::Move>;

    fn play(&mut self, action: &Self::Move);
    fn pass(&mut self);
    fn winner(&self) -> Option<Self::Player>;
    fn score(&self, player: Self::Player) -> f64;

    fn random_move(&mut self) -> (Self::GameHash, Option<Self::Move>) {
        let actions = self.possible_moves();
        let chosen_action = actions.choose(&mut rand::thread_rng());

        let gh = self.hash();

        match chosen_action {
            None => self.pass(),
            Some(action) => self.play(action),
        };
        (gh, chosen_action.map(|x| *x))
    }

    fn playout_board_history(&self) -> (Self, Vec<(Self::GameHash, Option<Self::Move>)>) {
        let mut s = *self;
        let mut hist = Vec::new();

        while {
            let v = s.random_move();
            hist.push(v);
            match s.winner() {
                None => true,
                Some(_) => false,
            }
        } {}
        (s, hist)
    }

    fn playout_history(&self) -> (Self::Player, Vec<(Self::GameHash, Option<Self::Move>)>) {
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

use cursive;

pub trait InteractiveGame: cursive::view::View {
    type G: Game;

    fn new(turn: <Self::G as Game>::Player) -> Self;

    fn get_mut(&mut self) -> &mut Self::G;
    fn get(&self) -> &Self::G;
    fn choose_move(
        &mut self,
        cb: Box<dyn FnOnce(<Self::G as Game>::Move, &mut Self)>,
    );
}
