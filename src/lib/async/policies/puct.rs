
use std::f32;
use std::iter::*;

use crate::game::{MultiplayerGame, BaseGame};
use crate::game;
use crate::policies::N_PLAYOUTS;
use super::{AsyncMultiplayerPolicyBuilder};
use super::mcts::{AsyncMCTSPolicy, WithAsyncMCTSPolicy};


use async_trait::async_trait;

use std::collections::HashMap;
use std::marker::PhantomData;

use float_ord::FloatOrd;

#[derive(Debug)]
pub struct PUCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
    pub pi: f32,
}

#[derive(Debug)]
pub struct PUCTNodeInfo<G: MultiplayerGame> {
    pub count: f32,
    pub moves: HashMap<G::Move, PUCTMoveInfo>,
}

use ndarray::Array;

pub trait FutureOutput<G:  game::Feature>: std::future::Future<Output=(Array<f32, G::ActionDim>, f32)> {}
impl<G: game::Feature, O: std::future::Future<Output=(Array<f32, G::ActionDim>, f32)>> FutureOutput<G> for O {}

pub trait AsyncEvaluator<G:  game::Feature, O:FutureOutput<G>>: Fn(G::Player, &[G]) -> O {}
impl<'a, G: game::Feature, O:FutureOutput<G>, F: Fn(G::Player, &[G]) -> O> AsyncEvaluator <G,O>for F {}

pub struct PUCTPolicy_<'a, G: game::Feature, O:FutureOutput<G>,F: AsyncEvaluator<G,O>> {
    pub color: G::Player,
    pub C_PUCT: f32,
    pub evaluate: &'a F,
    pub tree: HashMap<usize, PUCTNodeInfo<G>>,
    pub _o: PhantomData<O>,
}

#[async_trait]
impl<'a,'b, G, O, F> AsyncMCTSPolicy<G> for PUCTPolicy_<'a, G, O, F>
where
    G: game::Feature + Send + Sync,
    G::Player: Send + Sync,
    G::Move: Send + Sync,
    O: FutureOutput<G> + Send + Sync,
    F: AsyncEvaluator<G,O> + Sync
{
    type NodeInfo = PUCTNodeInfo<G>;
    type PlayoutInfo = (Option<HashMap<G::Move, f32>>, f32, G); // (policy, value, leaf_state).

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        &self.tree
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        &mut self.tree
    }

    fn select_move(&self, board: &G, _exploration: bool) -> G::Move {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board.hash()).unwrap();
        let N = node_info.count;

        let moves_scores = moves.iter().map(|action| {
            let v = node_info.moves.get(action).unwrap();
            let value = v.Q + self.C_PUCT * v.pi * (N.sqrt() / (v.N_a + 1.));
            (value, action)
        });

        if board.turn() == self.color {
            *moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        } else {
            *moves_scores.min_by_key(|x| FloatOrd(x.0)).unwrap().1
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        let n_moves = board.possible_moves().len();
        let moves = HashMap::from_iter(board.possible_moves().into_iter().map(|m| {
            (
                *m,
                PUCTMoveInfo {
                    Q: 0.,
                    N_a: 0.,
                    pi: 1. / (n_moves as f32),
                },
            )
        }));
        PUCTNodeInfo { count: 0., moves }
    }

    async fn simulate(&self, board: &G) -> Self::PlayoutInfo {
        if !board.is_finished() {
            let (policy, value) = (self.evaluate)(self.color, &[board.clone()]).await;
            let policy = board.feature_to_moves(&policy);
            (Some(policy), value, board.clone())
        } else {
            if board.has_won(self.color) {
                (None, 1., board.clone())
            } else {
                (None, 0., board.clone())
            }
        }
    }

    fn backpropagate(&mut self, history: Vec<(usize, G::Move)>, (policy, value, board): Self::PlayoutInfo) {
        if let Some(policy) = policy { // save probabilities of newly created node.
            let z: f32 = board
                .possible_moves()
                .into_iter()
                .map(|m| policy.get(&m).unwrap())
                .sum();
            let z = if z == 0. { 1. } else { z };
            
            for (m, info) in self.tree.get_mut(&board.hash()).unwrap().moves.iter_mut() {
                info.pi = policy.get(&m).unwrap() / z;
            }
            ;
        };

        let value = if board.turn() == self.color { value } else { 1. - value };

        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (value - (*v).Q) / (*v).N_a;
        }
    }
}


pub type PUCTPolicy<'a,G,O,F> = WithAsyncMCTSPolicy<G, PUCTPolicy_<'a,G,O,F>> ;


// POLICY BUILDER

pub struct PUCT<'a, G, O, F> 
    where 
    G: game::Feature + Send + Sync,
    O: FutureOutput<G> + Send + Sync,
    G::Player: Send + Sync,
    G::Move: Send + Sync,
    F: (Fn(G::Player, &[G]) -> O) + Sync
{
    pub C_PUCT: f32,
    pub evaluate: &'a F,
    pub _g: PhantomData<fn() -> G>,
    pub _o: PhantomData<fn() -> O>,
}

impl<G,O,F> Copy for PUCT<'_, G, O, F> 
where
    G: game::Feature + Send + Sync,
    G::Player: Send + Sync,
    G::Move: Send + Sync,
    O: FutureOutput<G> + Send + Sync,
    F: AsyncEvaluator<G,O> + Sync
{}

impl<G,O,F> Clone for PUCT<'_, G, O, F>
where
    G: game::Feature + Send + Sync,
    G::Player: Send + Sync,
    G::Move: Send + Sync,
    O: FutureOutput<G> + Send + Sync,
    F: AsyncEvaluator<G,O> + Sync
 {
    fn clone(&self) -> Self {
        *self
    }
}

use std::fmt;
impl<G,O,F> fmt::Display
    for PUCT<'_, G, O, F>
where
    G: game::Feature + Send + Sync,
    G::Player: Send + Sync,
    G::Move: Send + Sync,
    O: FutureOutput<G> + Send + Sync,
    F: AsyncEvaluator<G,O> + Sync
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PUCT")?;
        writeln!(f, "|| C_PUCT: {}", self.C_PUCT)?;
        writeln!(f, "|| N_PLAYOUTS: {}", N_PLAYOUTS)
    }
}

impl<'a, G, O, F> AsyncMultiplayerPolicyBuilder<G>for PUCT<'a, G, O, F>
where
    G: game::Feature + Send + Sync,
    G::Player: Send + Sync,
    G::Move: Send + Sync,
    O: FutureOutput<G> + Send + Sync,
    F: AsyncEvaluator<G,O> + Sync
{
    type P = PUCTPolicy<'a, G, O, F>;

    fn create(&self, color: G::Player) -> Self::P {
        WithAsyncMCTSPolicy::new(PUCTPolicy_::<'a, G, O, F> {
            color,
            C_PUCT: self.C_PUCT,
            evaluate: self.evaluate,
            tree: HashMap::new(),
            _o: PhantomData
        })
    }
}
