use rand::seq::SliceRandom;
use std::fmt::Debug;

use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

pub mod breakthrough;
pub mod misere_breakthrough;
pub mod weak_schur;
pub mod hashcode_20;

/** 
 * Common interface for single and multiplayer games
 */
pub trait BaseGame: Sized + Clone + Debug {
    /**
     * The type for a Move.
     */
    type Move: PartialEq + Eq + Copy + Clone + Hash + Debug;
    /**
     * Given the game state and turn, list possible actions.
     */
    fn possible_moves(&self) -> &[Self::Move];
    /**
     * Returns if the game has ended or not.
     */
    fn is_finished(&self) -> bool {
        return self.possible_moves().is_empty();
    }
    /**
     * Mutates game state playing the given action.
     */
    fn play(&mut self, action: &Self::Move);
    /**
     * Pseudo-unique value representing the game state.
     */
    fn hash(&self) -> usize;

    /**
     * Plays a random move or does nothing if there's no move to play. 
     */
    fn random_move(&mut self) -> (usize, Option<Self::Move>) {
        let actions = self.possible_moves();
        let chosen_action = actions.choose(&mut rand::thread_rng()).copied();

        let gh = self.hash();

        match chosen_action {
            None => (),
            Some(action) => self.play(&action),
        };
        (gh, chosen_action)
    }

    fn playout_full_history(&self) -> (Self, Vec<(Self, Self::Move)>) {
        let mut s = self.clone();
        let mut hist = Vec::new();

        while { !s.is_finished() } {
            let s_cloned = s.clone();
            let (_,m) = s.random_move();
            hist.push((s_cloned, m.unwrap()));
        }
        (s, hist)
    }

    fn playout_board_history(&self) -> (Self, Vec<(usize, Self::Move)>) {
        let mut s = self.clone();
        let mut hist = Vec::new();

        while { !s.is_finished() } {
            let (h,m) = s.random_move();
            hist.push((h, m.unwrap()));
        }
        (s, hist)
    }

    fn playout_board(&self) -> Self {
        let (s, _) = self.playout_board_history();
        s
    }
}

pub trait MultiplayerGameBuilder<G: MultiplayerGame> {
    fn create(&self, starting: G::Player) -> G;
}

pub trait MultiplayerGame: BaseGame {
    type Player: PartialEq + Eq + Copy + Clone + Debug;

    fn players() -> Vec<Self::Player>;    
    fn turn(&self) -> Self::Player;

    fn has_won(&self, player: Self::Player) -> bool;
}

pub trait SingleplayerGameBuilder<G: SingleplayerGame> {
    fn create(&self) -> G;
}

pub trait SingleplayerGame: BaseGame {
    fn score(&self) -> f32;
}

/* TODO: MoveCode */
pub trait MoveCode<G: BaseGame> {
    fn code(game: &G, action: &G::Move) -> usize;
}

pub struct NoFeatures {}
impl<T: BaseGame> MoveCode<T> for NoFeatures {
    fn code(_: &T, action: &T::Move) -> usize {
        let mut s = DefaultHasher::new();
        action.hash(&mut s);
        usize::try_from(s.finish()).unwrap()
    }
}

/* GAME WITH AN UI */
pub trait InteractiveGame: cursive::view::View {
    type G: MultiplayerGame;   

    fn new(turn: <Self::G as MultiplayerGame>::Player) -> Self;

    fn get_mut(&mut self) -> &mut Self::G;
    fn get(&self) -> &Self::G;
    fn choose_move(&mut self, cb: Box<dyn FnOnce(<Self::G as BaseGame>::Move, &mut Self)>);
}
