use crate::game::Game;
use crate::policies::MultiplayerPolicy;

use std::collections::HashMap;
use std::iter::*;
use std::hash::Hash;
use std::marker::PhantomData;
use async_trait::async_trait;

pub mod uct;
pub mod rave;
pub mod puct;

pub trait MCTSGame = Game + Clone + Hash + Eq;
/* ABSTRACT MCTS */

pub trait BaseMCTSPolicy<G: MCTSGame> {
    type NodeInfo;
    type PlayoutInfo;

    fn tree(&self) -> &HashMap<G, Self::NodeInfo>;
    fn tree_mut(&mut self) -> &mut HashMap<G, Self::NodeInfo>;

    fn select_move(&self, board: &G, exploration: bool) -> G::Move;

    fn select(&self, board: &mut G) -> Vec<(G, G::Move)> {
        let mut history: Vec<(G, G::Move)> = Vec::new();

        while !board.is_finished() {
            match self.tree().get(&board) {
                None => {
                    /* we're at a leaf node. */
                    return history;
                }
                Some(_node) => {
                    /* play next move */
                    let a = self.select_move(&board, true);
                    history.push((board.clone(), a));
                    board.play(&a)
                }
            };
        }
        history
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo;

    fn expand(&mut self, board: &G) {
        let new_node = self.default_node(board);
        self.tree_mut().insert(board.clone(), new_node);
    }

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, playout: Self::PlayoutInfo);

}


pub trait MCTSPolicy<G>: BaseMCTSPolicy<G>
where
    G: MCTSGame,
    G::Move: Send
{
    fn simulate(&self, board: &G) -> Self::PlayoutInfo;
    
    fn tree_search(&mut self, board: &G) {
        let mut b_scratch = board.clone();
        let selection_history = self.select(&mut b_scratch);
        self.expand(&b_scratch);
        let playout = self.simulate(&b_scratch);
        self.backpropagate(selection_history, playout);
    }
}

pub struct WithMCTSPolicy<G, M>
{
    pub inner: M, 
    N_PLAYOUTS: usize,
    _g: std::marker::PhantomData<G>,
}

impl<G, M> WithMCTSPolicy<G, M>
where
    G: MCTSGame + Clone,
    M: MCTSPolicy<G>,
{
    pub fn new(p: M, N_PLAYOUTS: usize) -> Self {
        WithMCTSPolicy {inner: p, N_PLAYOUTS, _g: PhantomData}
    }
}


impl<G, M> MultiplayerPolicy<G> for WithMCTSPolicy<G, M>
where
    G: MCTSGame,
    M: MCTSPolicy<G>,
{
    fn play(&mut self, board: &G) -> G::Move {
        self.inner.tree_mut().clear();

        for _ in 0..self.N_PLAYOUTS {
            self.inner.tree_search(board)
        }

        self.inner.select_move(board, false)
    }
}
