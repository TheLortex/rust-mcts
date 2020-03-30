use ndarray::{Array, Axis, Dimension};
use rand::seq::SliceRandom;
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::policies::MultiplayerPolicy;

/**
 *  Breakthrough game implementation
 *
 *  The rules are simple, a player wins when one of its pawns
 *  reaches the other side of the board.
 */
pub mod breakthrough;
/*pub mod hashcode_20;
pub mod misere_breakthrough;
pub mod weak_schur;*/
/**
 *  Games that takes other games as an input.
 */
pub mod meta;

/// Action that may be applied to a game state.
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
     * Given the game state and turn, list possible actions to the current player.
     * If the game has ended, no action should be available.
     */
    fn possible_moves(&self) -> Vec<Self::Move>;
    /**
     * Returns if the game has ended or not.
     */
    fn is_finished(&self) -> bool {
        self.possible_moves().is_empty()
    }
}

/**
 * Mutable game by playing moves.
 */
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

/**
 *  Game with one or multiple players.
 */
pub trait Game: Playable {
    /**
     *  The type representing each player.
     */
    type Player: PartialEq + Eq + Copy + Clone + Debug + Sync + Send + Into<u8>;

    /**
     *  Assuming a static player order, returns who should play after given player.
     */
    fn player_after(player: Self::Player) -> Self::Player;

    /**
     *  Returns the list of players for the game.
     */
    fn players() -> Vec<Self::Player>;

    /**
     *  Returns whose turn it is.
     */
    fn turn(&self) -> Self::Player;
}

/**
 *  Game playouts
 */
pub trait Playout: Game + Clone {
    /**
     *  Simulate a game execution using random moves until reaching a final state.
     *  It stores moves and state history, along with the total reward and the final state.
     */
    fn playout_history(&self, pov: Self::Player) -> (Self, Vec<(Self, Self::Move)>, f32) {
        let mut s = self.clone();
        let mut hist = Vec::new();

        let mut total_reward = 0.;

        while { !s.is_finished() } {
            let s_cloned = s.clone();
            let player = s.turn();
            let (m, r) = s.random_move();
            if player == pov {
                total_reward += r;
            }

            hist.push((s_cloned, m));
        }
        (s, hist, total_reward)
    }

    /**
     *  Simulates a game execution using random moves until reaching a final state.
     *  It returns the total reward with the final state.
     */
    fn playout_board(&self, pov: Self::Player) -> (Self, f32) {
        let (s, _, total_reward) = self.playout_history(pov);
        (s, total_reward)
    }
}
impl<G: Game + Clone> Playout for G {}

/**
 *  Non-cooperative games.
 */
pub trait SingleWinner: Game {
    /// Returns the winner of the game, or None if no one has won yet.
    fn winner(&self) -> Option<Self::Player>;
}

/**
 *  Single-player games.
 */
pub trait Singleplayer: Playable {}

/**
 *  A single-player game can be written as a game with a default player
 *  that plays all the turns.
 */
impl<G: Singleplayer> Game for G {
    type Player = u8;

    fn players() -> Vec<Self::Player> {
        vec![0]
    }

    fn player_after(_player: Self::Player) -> Self::Player {
        0
    }

    fn turn(&self) -> Self::Player {
        0
    }
}

/**
 *  Game builders.
 */
pub trait GameBuilder<G: Game> {
    /**
     *  Create a new game starting for player `starting`.
     */
    fn create(&self, starting: G::Player) -> G;
}

/**
 *  Builders for single-player games.
 */
pub trait SingleplayerGameBuilder<G: Singleplayer> {
    /**
     *  Create a new single-player game instance.
     */
    fn create(&self) -> G;
}
/**
 *  Single-player game builder is an instance of GameBuilder
 */
impl<G: Singleplayer, GB: SingleplayerGameBuilder<G>> GameBuilder<G> for GB {
    fn create(&self, _starting: <G as Game>::Player) -> G {
        self.create()
    }
}

use std::collections::HashMap;
/**
 * Games that can be represented as multi-dimensional arrays.
 *
 * These games are the ones that can be played by neural network-based policies,
 * such as PUCT or Muz.
 */
pub trait Feature: Game {
    /**
     *  Type dimension of the game state feature space.
     */
    type StateDim: Dimension;
    /**
     *  Type dimension of the action feature space.
     */
    type ActionDim: Dimension;

    /**
     *  Game state dimension.
     */
    fn state_dimension(&self) -> Self::StateDim;
    /**
     *  Action space dimension.
     */
    fn action_dimension() -> Self::ActionDim;

    /**
     *  Converts the game state to features (multi-dimensional array).
     *
     *  These features may be relative to a particular player but they should
     *  contain the same amount of information.
     */
    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim>;

    /**
     *  Converts an action probability distribution to the action features.
     */
    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim>;

    /**
     *  Converts a single move to a one-hot feature encoding of the move.
     */
    fn move_to_feature(action: Self::Move) -> Array<f32, Self::ActionDim> {
        let mut hash = HashMap::new();
        hash.insert(action, 1.);
        Self::moves_to_feature(&hash)
    }
    /**
     *  Converts a move distribution feature to the corresponding set of move probabilities, relative to the game state.
     *  
     *  Invalid moves relative to the game state are discarded. To keep all moves, see
     *  `all_feature_to_moves`.
     */
    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32>;

    /**
     *  Returns action space
     */
    fn all_possible_moves() -> Vec<Self::Move>;

    /**
     *  Converts a move distribution feature to the corresponding set of move probabilities, independently from the game state.
     *
     *  To consider only valid moves, see `feature_to_moves`.
     */
    fn all_feature_to_moves(features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32>;
}

/**
 *  Move encoders
 */
pub trait MoveCode<G: Base> {
    /**
     *  Encode a move given the current state.
     */
    fn code(game: &G, action: &G::Move) -> usize;
}

/**
 *  A move encoder that doesn't take into account the
 *  board state.
 */
pub struct NoFeatures {}
impl<T: Base> MoveCode<T> for NoFeatures {
    fn code(_: &T, action: &T::Move) -> usize {
        let mut s = DefaultHasher::new();
        action.hash(&mut s);
        usize::try_from(s.finish()).unwrap()
    }
}

/**
 *  Games with an user interface.
 *
 *  Terminal user interface is managed by the `cursive` library.
 *
 */
pub trait InteractiveGame: cursive::view::View {
    /**
     *  Interfaced game type
     */
    type G: Game;

    /**
     *  Create a new instance of the UI.
     */
    fn new(turn: <Self::G as Game>::Player) -> Self;

    /**
     *  Play a chosen move.
     */
    fn play(&mut self, action: &<Self::G as Base>::Move);

    /**
     *  Retrieve internal game state.
     */
    fn get(&self) -> &Self::G;
}

/**
 *  Execute two policies on a two-player game
 *  and returns the whole history.
 */
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
        //println!("{:?} => {:?}", action, board);
        let game_has_ended = board.is_finished();
        history.push(board);
        !game_has_ended
    } {}
    history
}
