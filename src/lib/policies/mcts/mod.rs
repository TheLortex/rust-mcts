use crate::game::MultiplayerGame;
use crate::policies::{MultiplayerPolicy, AsyncMultiplayerPolicy};

use std::collections::HashMap;
use std::iter::*;
use std::marker::PhantomData;
use async_trait::async_trait;

pub mod uct;
pub mod rave;
pub mod puct;

/* ABSTRACT MCTS */
pub trait BaseMCTSPolicy<G: MultiplayerGame> {
    type NodeInfo;
    type PlayoutInfo;

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo>;
    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo>;

    fn select_move(&self, board: &G, exploration: bool) -> G::Move;

    fn select(&self, board: &mut G) -> Vec<(G, G::Move)> {
        let mut history: Vec<(G, G::Move)> = Vec::new();

        while !board.is_finished() {
            let s_t = board.hash();
            match self.tree().get(&s_t) {
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
        self.tree_mut().insert(board.hash(), new_node);
    }

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, playout: Self::PlayoutInfo);

}

/**
 *  SYNCHRONOUS MCTS
 */

pub trait MCTSPolicy<G: MultiplayerGame>: BaseMCTSPolicy<G> {
    fn simulate(&self, history: &[G]) -> Self::PlayoutInfo;

    fn tree_search(&mut self, history: &[G]) {
        let mut b = history.last().unwrap().clone();
        let selection_history = self.select(&mut b);

        let mut game_history = Vec::from(history);
        game_history.extend(selection_history.iter().map(|(g,_)| g.clone()));
        game_history.push(b.clone());
        self.expand(&b);
        let playout = self.simulate(&game_history);
        self.backpropagate(selection_history, playout);
    }
}

pub struct WithMCTSPolicy<G: MultiplayerGame, M: MCTSPolicy<G>> {
    pub inner: M, 
    N_PLAYOUTS: usize, 
    _g: std::marker::PhantomData<G>
}

impl<G: MultiplayerGame, M: MCTSPolicy<G>> WithMCTSPolicy<G, M> {
    pub fn new(p: M, N_PLAYOUTS: usize) -> Self {
        WithMCTSPolicy {
            inner: p, 
            N_PLAYOUTS, 
            _g: PhantomData
        }
    }
}

impl<G: MultiplayerGame, M: MCTSPolicy<G>> MultiplayerPolicy<G> for WithMCTSPolicy<G, M> {
    fn play(&mut self, history: &[G]) -> G::Move {
        self.inner.tree_mut().clear();
        let board = history.last().unwrap();
        for _ in 0..self.N_PLAYOUTS {
            self.inner.tree_search(history)
        }

        self.inner.select_move(board, false)
    }
}

/**
 * ASYNCHRONOUS VERSION:
 */
#[async_trait]
pub trait AsyncMCTSPolicy<G>: BaseMCTSPolicy<G> + Sync
where
    G: MultiplayerGame + Send + Sync,
    G::Move: Send
{
    async fn simulate(&self, history: &[G]) -> Self::PlayoutInfo;
    
    async fn tree_search(&mut self, history: &[G]) {
        let mut b = history.last().unwrap().clone();
        let selection_history = self.select(&mut b);

        let mut game_history = Vec::from(history);
        game_history.extend(selection_history.iter().map(|(g,_)| g.clone()));
        self.expand(&b);
        let playout = self.simulate(&game_history).await;
        self.backpropagate(selection_history, playout);
    }
}

pub struct WithAsyncMCTSPolicy<G, M>
{
    pub inner: M, 
    N_PLAYOUTS: usize,
    _g: std::marker::PhantomData<G>,
}

impl<G, M> WithAsyncMCTSPolicy<G, M>
where
    G: MultiplayerGame + Send + Sync,
    M: AsyncMCTSPolicy<G> + Send,
    G::Move: Send
{
    pub fn new(p: M, N_PLAYOUTS: usize) -> Self {
        WithAsyncMCTSPolicy {inner: p, N_PLAYOUTS, _g: PhantomData}
    }
}

#[async_trait]
impl<G, M> AsyncMultiplayerPolicy<G> for WithAsyncMCTSPolicy<G, M>
where
    G: MultiplayerGame + Send + Sync,
    M: AsyncMCTSPolicy<G> + Send,
    G::Move: Send
{
    async fn play(&mut self, history: &[G]) -> G::Move {
        self.inner.tree_mut().clear();
        let board = history.last().unwrap();

        for _ in 0..self.N_PLAYOUTS {
            self.inner.tree_search(history).await
        }

        self.inner.select_move(board, false)
    }
}
