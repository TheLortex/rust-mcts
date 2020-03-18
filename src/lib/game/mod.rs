use rand::seq::SliceRandom;
use std::fmt::Debug;
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use ndarray::{Array, Dimension};

pub mod breakthrough;
pub mod misere_breakthrough;
pub mod weak_schur;
pub mod hashcode_20;

pub trait MoveTrait = PartialEq + Eq + Copy + Clone + Hash + Debug;

/** 
 * Common interface for single and multiplayer games
 */
pub trait BaseGame: Sized + Clone + Debug {
    /**
     * The type for a Move.
     */
    type Move: MoveTrait;
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

use std::collections::HashMap;
/* FeatureGame */
pub trait Feature: MultiplayerGame {
    type StateDim: Dimension;
    type ActionDim: Dimension;

    fn state_dimension() -> Self::StateDim;
    fn action_dimension() -> Self::ActionDim;

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim>;
    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim>;

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32>;
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


use crate::policies::MultiplayerPolicy;

pub fn simulate<'a, 'b, G: MultiplayerGame>(
    mut p1: Box<dyn MultiplayerPolicy<G> + 'a>,
    mut p2: Box<dyn MultiplayerPolicy<G> + 'b>,
    game: &G,
) -> Vec<G> {
    let mut history = vec![game.clone()];

    while {
        let mut board = history.last().unwrap().clone();
        let action = if board.turn() == G::players()[0] {
            p1.play(&history)
        } else {
            p2.play(&history)
        };
        board.play(&action);
        let game_has_ended = board.is_finished();
        history.push(board);
        !game_has_ended
    } {}
    history
}