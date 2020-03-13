use std::collections::HashMap;


use super::super::game::MultiplayerGame;
use super::{AsyncMultiplayerPolicy, N_PLAYOUTS};

use std::marker::PhantomData;

/* ABSTRACT MCTS */

use async_trait::async_trait;

#[async_trait]
pub trait AsyncMCTSPolicy<G>: Sync
    where 
        G: MultiplayerGame + Sync + Send,
        G::Move: Sync + Send
     
{
    type NodeInfo;
    type PlayoutInfo;

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo>;
    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo>;

    fn select_move(&self, board: &G, exploration: bool) -> G::Move;

    fn select(&self, board: &mut G) -> Vec<(usize, G::Move)> {
        let mut history: Vec<(usize, G::Move)> = Vec::new();

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
                    history.push((s_t, a));
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

    async fn simulate(&self, board: &G) -> Self::PlayoutInfo;

    fn backpropagate(&mut self, history: Vec<(usize, G::Move)>, playout: Self::PlayoutInfo);

    async fn tree_search(&mut self, board: &G) {
        let mut b = board.clone();
        let history = self.select(&mut b);
        self.expand(&b);
        let playout = self.simulate(&b).await;
        self.backpropagate(history, playout);
    }
}

pub struct WithAsyncMCTSPolicy<G, M>
where
    G: MultiplayerGame + Sync + Send,
    M: AsyncMCTSPolicy<G>,
    G::Move: Sync + Send,
{
    pub inner: M, 
    _g: std::marker::PhantomData<G>
}

impl<G, M> WithAsyncMCTSPolicy<G, M>
where
    G: MultiplayerGame + Sync + Send,
    M: AsyncMCTSPolicy<G>,
    G::Move: Sync + Send,
{
    pub fn new(p: M) -> Self {
        WithAsyncMCTSPolicy {inner: p, _g: PhantomData}
    }
}

#[async_trait]
impl<G: MultiplayerGame, M: AsyncMCTSPolicy<G>> AsyncMultiplayerPolicy<G> for WithAsyncMCTSPolicy<G, M>
where
    G: MultiplayerGame,
    M: AsyncMCTSPolicy<G> + Sync + Send,
    G::Move: Sync + Send,
    G: Sync + Send 
{
    async fn play(&mut self, board: &G) -> G::Move {
        for _ in 0..N_PLAYOUTS {
            self.inner.tree_search(board).await
        }

        self.inner.select_move(&board, false)
    }
}
