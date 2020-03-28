use ndarray::{Array, Axis, Dimension};
use rand::seq::SliceRandom;
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};



pub mod breakthrough;
/*pub mod hashcode_20;
pub mod misere_breakthrough;
pub mod weak_schur;*/
pub mod meta;

pub trait MoveTrait = PartialEq + Eq + Copy + Clone + Hash + Debug;

/**
 * Common interface for single and multiplayer games
 */
pub trait Base: Sized + Debug {
    /**
     * The type for a Move.
     */
    type Move: MoveTrait;
    /**
     * Given the game state and turn, list possible actions.
     */
    fn possible_moves(&self) -> Vec<Self::Move>;
    /**
     * Returns if the game has ended or not.
     */
    fn is_finished(&self) -> bool {
        self.possible_moves().is_empty()
    }
}


pub trait Playable: Base {
    /**
     * Mutates game state playing the given action. 
     * Yields a reward to the player.
     */
    fn play(&mut self, action: &Self::Move) -> f32;

    /**
     * Plays a random move. Yields a reward.
     */
    fn random_move(&mut self) -> (Self::Move, f32) {
        let actions = self.possible_moves();
        let chosen_action = actions.choose(&mut rand::thread_rng()).unwrap();
        let reward = self.play(chosen_action);
        (*chosen_action, reward)
    }
}


pub trait Playout: Game + Clone {
    fn playout_history(&self, pov: Self::Player) -> (Self, Vec<(Self, Self::Move)>, f32) {
        let mut s = self.clone();
        let mut hist = Vec::new();

        let mut total_reward = 0.;

        while { !s.is_finished() } {
            let s_cloned = s.clone();
            let player = s.turn();
            let (m,r) = s.random_move();
            if player == pov {
                total_reward += r;
            }

            hist.push((s_cloned, m));
        }
        (s, hist, total_reward)
    }

    fn playout_board(&self, pov: Self::Player) -> (Self, f32) {
        let (s, _, total_reward) = self.playout_history(pov);
        (s, total_reward)
    }
}
impl<G: Game + Clone> Playout for G {}

pub trait Game: Playable {
    type Player: PartialEq + Eq + Copy + Clone + Debug + Sync + Send + Into<u8>;


    fn player_after(player: Self::Player) -> Self::Player;
    fn players() -> Vec<Self::Player>;
    fn turn(&self) -> Self::Player;
}

pub trait SingleWinner: Game {
    fn winner(&self) -> Option<Self::Player>;
}

pub trait Singleplayer: Playable {}

impl<G: Singleplayer> Game for G {
    type Player = u8;

    fn players() -> Vec<Self::Player> {
        vec![0]
    }

    fn player_after(_player: Self::Player) -> Self::Player {0}

    fn turn(&self) -> Self::Player {0}
}

pub trait GameBuilder<G: Game> {
    fn create(&self, starting: G::Player) -> G;
}

pub trait SingleplayerGameBuilder<G: Singleplayer> {
    fn create(&self) -> G;
}
impl<G: Singleplayer, GB: SingleplayerGameBuilder<G>> GameBuilder<G> for GB {
    fn create(&self, _starting: <G as Game>::Player) -> G {
        self.create()
    }
}

use std::collections::HashMap;
/* FeatureGame */
pub trait Feature: Game {
    type StateDim: Dimension;
    type ActionDim: Dimension;

    fn state_dimension(&self) -> Self::StateDim;
    fn action_dimension() -> Self::ActionDim;

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim>;
    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim>;

    fn move_to_feature(action: Self::Move) -> Array<f32, Self::ActionDim> {
        let mut hash = HashMap::new();
        hash.insert(action, 1.);
        Self::moves_to_feature(&hash)
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32>;

    fn all_possible_moves() -> Vec<Self::Move>;
    fn all_feature_to_moves(features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32>;
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

    fn play(&mut self, action: &<Self::G as Base>::Move);
    fn get(&self) -> &Self::G;
   //TODO fn choose_move(&mut self, cb: Box<dyn FnOnce(<Self::G as Base>::Move, &mut Self)>);
}

use crate::policies::MultiplayerPolicy;

pub fn simulate<'a, 'b, G: Game + Clone>(
    mut p1: Box<dyn MultiplayerPolicy<G> + 'a>,
    mut p2: Box<dyn MultiplayerPolicy<G> + 'b>,
    game: &G,
) -> Vec<G> 
{
    let mut history = vec![game.clone()];

    while {
        let mut board = history.last().unwrap().clone();
        let action = if board.turn() == G::players()[0] {
            p1.play(&board)
        } else {
            p2.play(&board)
        };
        board.play(&action);
        //println!("{:?} => {:?}", action, board);
        let game_has_ended = board.is_finished();
        history.push(board);
        !game_has_ended
    } {}
    history
}
