use crate::game;
use crate::policies::mcts::{
    AsyncMCTSPolicy, BaseMCTSPolicy, MCTSPolicy, WithAsyncMCTSPolicy, WithMCTSPolicy,
};
use crate::policies::{AsyncMultiplayerPolicyBuilder, MultiplayerPolicyBuilder};

use async_trait::async_trait;
use float_ord::FloatOrd;
use ndarray::Array;
use std::collections::HashMap;
use std::f32;
use std::future::Future;
use std::iter::*;
use std::marker::PhantomData;
use rand_distr::{Distribution, Gamma};


#[derive(Debug)]
pub struct PUCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
    pub pi: f32,
}

#[derive(Debug)]
pub struct PUCTNodeInfo<G: game::Feature> {
    pub count: f32,
    pub moves: HashMap<G::Move, PUCTMoveInfo>,
}
/**
 * The game state evaluator
 */
pub trait Evaluator<G: game::Feature>:
    Fn(G::Player, &[G]) -> (Array<f32, G::ActionDim>, f32)
{
}
impl<G: game::Feature, F: Fn(G::Player, &[G]) -> (Array<f32, G::ActionDim>, f32)> Evaluator<G>
    for F
{
}
/**
 * The game state evaluator (async edition)
 */
pub trait FutureOutput<G>: Future<Output = (Array<f32, G::ActionDim>, f32)>
where
    G: game::Feature,
{
}

impl<G, O> FutureOutput<G> for O
where
    G: game::Feature,
    O: Future<Output = (Array<f32, G::ActionDim>, f32)>,
{
}

pub trait AsyncEvaluator<G, O>: Fn(G::Player, &[G]) -> O
where
    G: game::Feature,
    O: FutureOutput<G>,
{
}

impl<'a, G, O, F> AsyncEvaluator<G, O> for F
where
    G: game::Feature,
    O: FutureOutput<G>,
    F: Fn(G::Player, &[G]) -> O,
{
}
/**
 * Common PUCT
 */
#[derive(Copy, Clone, fmt::Debug)]
pub struct PUCTSettings {
    pub C_BASE: f32,
    pub C_INIT: f32,
    pub ROOT_DIRICHLET_ALPHA: f32,
    pub ROOT_EXPLORATION_FRACTION: f32,
    pub N_HISTORY: usize,
}

/**
 * Default inspired from AlphaZero paper.
 */
impl Default for PUCTSettings {
    fn default() -> Self {
        Self {
            C_BASE: 19652.,
            C_INIT: 1.25,
            ROOT_DIRICHLET_ALPHA: 0.3,
            ROOT_EXPLORATION_FRACTION: 0.25,
            N_HISTORY: 2,
        }
    }
}

pub struct BasePUCTPolicy_<G: game::Feature> {
    pub color: G::Player,
    pub s: PUCTSettings,
    pub tree: HashMap<usize, PUCTNodeInfo<G>>,
}

impl<G> BaseMCTSPolicy<G> for BasePUCTPolicy_<G>
where
    G: game::Feature,
{
    type NodeInfo = PUCTNodeInfo<G>;
    type PlayoutInfo = (Option<HashMap<G::Move, f32>>, f32, G);
    //                 (policy, value, leaf_state).

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
            let pb_c = ((N + self.s.C_BASE + 1.)/self.s.C_BASE).ln() + self.s.C_INIT;
            let value = v.Q + pb_c * v.pi * (N.sqrt() / (v.N_a + 1.));
            (value, action)
        });

        *moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
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

    fn backpropagate(
        &mut self,
        history: Vec<(G, G::Move)>,
        (policy, value, board): Self::PlayoutInfo,
    ) {
        if let Some(mut policy) = policy {
            // save probabilities of newly created node.
            if self.tree.len() == 0 {
                // add dirichlet noise on the root node.
                let frac = self.s.ROOT_EXPLORATION_FRACTION;
                let gamma = Gamma::new(self.s.ROOT_DIRICHLET_ALPHA.into(), 1.0).unwrap();
                for (_, val) in policy.iter_mut() {
                    let noise = gamma.sample(&mut rand::thread_rng());
                    *val = frac * (*val) + (1. - frac) * noise;
                }
            }
            let z: f32 = board
                .possible_moves()
                .into_iter()
                .map(|m| policy.get(&m).unwrap())
                .sum();
            let z = if z == 0. { 1. } else { z };
            for (m, info) in self.tree.get_mut(&board.hash()).unwrap().moves.iter_mut() {
                info.pi = policy.get(&m).unwrap() / z;
            }
        };

        let pov = board.turn();

        for (state, action) in history.iter() {
            // reverse value each step back as we switch 
            // point of view.
            let value = if state.turn() == pov {
                value
            } else {
                1. - value
            };

            let mut node = self.tree.get_mut(&state.hash()).unwrap();
            node.count += 1.;
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (value - (*v).Q) / (*v).N_a;
        }
    }
}
/**
 * sync PUCT
 */

pub struct PUCTPolicy_<'a, G, F>
where
    G: game::Feature,
    F: Evaluator<G>,
{
    pub b: BasePUCTPolicy_<G>,
    pub evaluate: &'a F,
}

impl<'a, G, F> BaseMCTSPolicy<G> for PUCTPolicy_<'a, G, F>
where
    G: game::Feature,
    F: Evaluator<G>,
{
    type NodeInfo = <BasePUCTPolicy_<G> as BaseMCTSPolicy<G>>::NodeInfo;
    type PlayoutInfo = <BasePUCTPolicy_<G> as BaseMCTSPolicy<G>>::PlayoutInfo;

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        self.b.tree()
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        self.b.tree_mut()
    }

    fn select_move(&self, board: &G, expl: bool) -> G::Move {
        self.b.select_move(board, expl)
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        self.b.default_node(board)
    }

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, pi: Self::PlayoutInfo) {
        self.b.backpropagate(history, pi)
    }
}

impl<'a, G, F> MCTSPolicy<G> for PUCTPolicy_<'a, G, F>
where
    G: game::Feature,
    F: Evaluator<G>,
{
    fn simulate(&self, history: &[G]) -> Self::PlayoutInfo {
        let board = history.last().unwrap();
        if !board.is_finished() {
            let history: Vec<G> = if history.len() < self.b.s.N_HISTORY {
                let mut _h =
                    vec![history.first().unwrap().clone(); self.b.s.N_HISTORY - history.len()];
                _h.extend(history.iter().cloned());
                _h
            } else {
                Vec::from(&history[(history.len() - self.b.s.N_HISTORY)..(history.len())])
            };
            // NN predicts a good policy for current player 
            //  + expectation of winning from this state.
            let (policy, value) = (self.evaluate)(board.turn(), &history);
            let policy = board.feature_to_moves(&policy);
            (Some(policy), value, board.clone())
        } else {
            if board.has_won(self.b.color) {
                (None, 1., board.clone())
            } else {
                (None, 0., board.clone())
            }
        }
    }
}
/**
 * async PUCT
 */
pub struct BatchedPUCTPolicy_<'a, G, O, F>
where
    G: game::Feature,
    O: FutureOutput<G>,
    F: AsyncEvaluator<G, O>,
{
    pub b: BasePUCTPolicy_<G>,
    pub evaluate: &'a F,
    pub _o: PhantomData<O>,
}

impl<'a, G, O, F> BaseMCTSPolicy<G> for BatchedPUCTPolicy_<'a, G, O, F>
where
    G: game::Feature,
    O: FutureOutput<G>,
    F: AsyncEvaluator<G, O>,
{
    type NodeInfo = <BasePUCTPolicy_<G> as BaseMCTSPolicy<G>>::NodeInfo;
    type PlayoutInfo = <BasePUCTPolicy_<G> as BaseMCTSPolicy<G>>::PlayoutInfo;

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        self.b.tree()
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        self.b.tree_mut()
    }

    fn select_move(&self, board: &G, expl: bool) -> G::Move {
        self.b.select_move(board, expl)
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        self.b.default_node(board)
    }

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, pi: Self::PlayoutInfo) {
        self.b.backpropagate(history, pi)
    }
}

#[async_trait]
impl<'a, G, O, F> AsyncMCTSPolicy<G> for BatchedPUCTPolicy_<'a, G, O, F>
where
    G: game::Feature + Send + Sync,
    O: FutureOutput<G> + Sync + Send,
    F: AsyncEvaluator<G, O> + Sync,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
{
    async fn simulate(&self, history: &[G]) -> <Self as BaseMCTSPolicy<G>>::PlayoutInfo {
        let board = history.last().unwrap();
        if !board.is_finished() {
            let history: Vec<G> = if history.len() < self.b.s.N_HISTORY {
                let mut _h =
                    vec![history.first().unwrap().clone(); self.b.s.N_HISTORY - history.len()];
                _h.extend(history.iter().cloned());
                _h
            } else {
                Vec::from(&history[(history.len() - self.b.s.N_HISTORY)..(history.len())])
            };
            // NN predicts a good policy for current player + expectation of winning from this state.
            let (policy, value) = (self.evaluate)(board.turn(), &history).await;
            let policy = board.feature_to_moves(&policy);
            (Some(policy), value, board.clone())
        } else {
            if board.has_won(self.b.color) {
                (None, 1., board.clone())
            } else {
                (None, 0., board.clone())
            }
        }
    }
}

/**
 *  POLICY BUILDERS - SYNC
 */
pub type PUCTPolicy<'a, G, F> = WithMCTSPolicy<G, PUCTPolicy_<'a, G, F>>;

#[derive(Copy, Clone)]
pub struct PUCT<'a, G, F>
where
    G: game::Feature,
    F: Fn(G::Player, &[G]) -> (Array<f32, G::ActionDim>, f32),
{
    pub s: PUCTSettings,
    pub N_PLAYOUTS: usize,
    pub evaluate: &'a F,
    pub _g: PhantomData<fn() -> G>,
}
/*
impl<G: game::Feature, F: Evaluator<G>> Copy for PUCT<'_, G, F> {}

impl<G: game::Feature, F: Evaluator<G>> Clone for PUCT<'_, G, F> {
    fn clone(&self) -> Self {
        *self
    }
}
*/
use std::fmt;
impl<G: game::Feature, F: Evaluator<G>> fmt::Display for PUCT<'_, G, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SYNC PUCT")?;
        writeln!(f, "||{:?}", self.s)
    }
}

impl<'a, G, F> MultiplayerPolicyBuilder<G> for PUCT<'a, G, F>
where
    G: game::Feature,
    F: Evaluator<G>,
{
    type P = PUCTPolicy<'a, G, F>;

    fn create(&self, color: G::Player) -> PUCTPolicy<'a, G, F> {
        WithMCTSPolicy::new(
            PUCTPolicy_::<'a, G, F> {
                b: BasePUCTPolicy_::<G> {
                    color,
                    s: self.s,
                    tree: HashMap::new(),
                },
                evaluate: self.evaluate,
            },
            self.N_PLAYOUTS,
        )
    }
}

/**
 *  POLICY BUILDERS - ASYNC
 */
pub type BatchedPUCTPolicy<'a, G, O, F> = WithAsyncMCTSPolicy<G, BatchedPUCTPolicy_<'a, G, O, F>>;

pub struct BatchedPUCT<'a, G, O, F>
where
    G: game::Feature,
    O: FutureOutput<G>,
    F: (Fn(G::Player, &[G]) -> O),
{
    pub s: PUCTSettings,
    pub N_PLAYOUTS: usize,
    pub evaluate: &'a F,
    pub _g: PhantomData<fn() -> G>,
}

impl<G, O, F> fmt::Display for BatchedPUCT<'_, G, O, F>
where
    G: game::Feature,
    O: FutureOutput<G>,
    F: (Fn(G::Player, &[G]) -> O),
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BATCHED PUCT")?;
        writeln!(f, "||{:?}", self.s)
    }
}

impl<'a, G, O, F> AsyncMultiplayerPolicyBuilder<G> for BatchedPUCT<'a, G, O, F>
where
    G: game::Feature + Send + Sync,
    O: FutureOutput<G> + Sync + Send,
    G::Move: Send + Sync,
    G::Player: Send + Sync,
    F: (Fn(G::Player, &[G]) -> O) + Sync,
{
    type P = BatchedPUCTPolicy<'a, G, O, F>;

    fn create(&self, color: G::Player) -> BatchedPUCTPolicy<'a, G, O, F> {
        WithAsyncMCTSPolicy::new(
            BatchedPUCTPolicy_::<'a, G, O, F> {
                b: BasePUCTPolicy_::<G> {
                    color,
                    s: self.s,
                    tree: HashMap::new(),
                },
                evaluate: self.evaluate,
                _o: PhantomData,
            },
            self.N_PLAYOUTS,
        )
    }
}