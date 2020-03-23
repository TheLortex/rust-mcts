use ndarray::{Array, Axis, Dimension};
use rand::seq::SliceRandom;
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;

pub mod breakthrough;
/*pub mod hashcode_20;
pub mod misere_breakthrough;
pub mod weak_schur;*/
pub mod meta;

pub trait MoveTrait = PartialEq + Eq + Copy + Clone + Hash + Debug + Send + Sync;

/**
 * Common interface for single and multiplayer games
 */
pub trait Base: Sized + Debug + Send + Sync {
    /**
     * The type for a Move.
     */
    type Move: MoveTrait;

    type MoveIterator<'a>: Iterator<Item=Self::Move> + Send;
    /**
     * Given the game state and turn, list possible actions.
     */
    fn possible_moves<'a>(&'a self) -> Self::MoveIterator<'a>;
    /**
     * Returns if the game has ended or not.
     */
    fn is_finished(&self) -> bool {
        self.possible_moves().next().is_none()
    }
}


pub trait Playable: Base {
    /**
     * Mutates game state playing the given action.
     */
    fn play(&mut self, action: &Self::Move);

    /**
     * Plays a random move or does nothing if there's no move to play.
     */
    fn random_move(&mut self) -> Option<Self::Move> {
        let actions = self.possible_moves();
        let chosen_action = actions.collect::<Vec<Self::Move>>().choose(&mut rand::thread_rng()).copied();

        match chosen_action {
            None => (),
            Some(action) => self.play(&action),
        };
        chosen_action
    }
}


pub trait Playout: Playable + Clone {
    fn playout_history(&self) -> (Self, Vec<(Self, Self::Move)>) {
        let mut s = self.clone();
        let mut hist = Vec::new();

        while { !s.is_finished() } {
            let s_cloned = s.clone();
            let m = s.random_move();
            hist.push((s_cloned, m.unwrap()));
        }
        (s, hist)
    }

    fn playout_board(&self) -> Self {
        let (s, _) = self.playout_history();
        s
    }
}
impl<G: Base + Playable + Clone> Playout for G {}

pub trait Game: Playable {
    type Player: PartialEq + Eq + Copy + Clone + Debug + Sync + Send;

    fn players() -> Vec<Self::Player>;
    fn turn(&self) -> Self::Player;

    fn has_won(&self, player: Self::Player) -> bool;
}

pub trait SingleWinner: Game {
    fn winner(&self) -> Option<Self::Player>;
}

pub trait Reward: Game {
    fn reward(&self, player: Self::Player) -> f32;
}

pub trait Singleplayer: Playable {
    fn score(&self) -> f32;
}

impl<G: Singleplayer> Game for G {
    type Player = ();

    fn players() -> Vec<Self::Player> {
        vec![()]
    }

    fn turn(&self) -> Self::Player {
        ()
    }

    fn has_won(&self, _player: Self::Player) -> bool {
        self.is_finished()
    }
}

impl<G: Singleplayer> Reward for G {
    fn reward(&self, _player: Self::Player) -> f32 {
        self.score()
    }
}

pub trait GameBuilder<G: Game> {
    fn create(&self, starting: G::Player) -> G;
}

pub trait SingleplayerGameBuilder<G: Singleplayer> {
    fn create(&self) -> G;
}
impl<G: Singleplayer, GB: SingleplayerGameBuilder<G>> GameBuilder<G> for GB {
    fn create(&self, starting: <G as Game>::Player) -> G {
        self.create()
    }
}

use std::collections::HashMap;
/* FeatureGame */
pub trait Feature: Game {
    type StateDim: Dimension;
    type ActionDim: Dimension;

    fn state_dimension() -> Self::StateDim;
    fn action_dimension() -> Self::ActionDim;

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim>;
    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim>;

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32>;
}

/* TODO: MoveCode */
pub trait MoveCode<G: Base> {
    fn code(game: &G, action: &G::Move) -> usize;
}

pub struct NoFeatures {}
impl<T: Base> MoveCode<T> for NoFeatures {
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
    fn choose_move(&mut self, cb: Box<dyn FnOnce(<Self::G as Base>::Move, &mut Self)>);
}

use crate::policies::MultiplayerPolicy;

pub fn simulate<'a, 'b, G: Game + Clone>(
    mut p1: Box<dyn MultiplayerPolicy<G> + 'a>,
    mut p2: Box<dyn MultiplayerPolicy<G> + 'b>,
    game: &G,
) -> Vec<G> {
    let mut history = vec![game.clone()];

    while {
        let mut board = history.last().unwrap().clone();
        let action = if board.turn() == G::players()[0] {
            p1.play(&board)
        } else {
            p2.play(&board)
        };
        board.play(&action);
        let game_has_ended = board.is_finished();
        history.push(board);
        !game_has_ended
    } {}
    history
}
