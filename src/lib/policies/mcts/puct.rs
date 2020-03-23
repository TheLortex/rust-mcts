use crate::game;
use crate::policies::mcts::{
    MCTSPolicy, BaseMCTSPolicy, WithMCTSPolicy,
};
use crate::policies::{MultiplayerPolicyBuilder};

use async_trait::async_trait;
use float_ord::FloatOrd;
use ndarray::Array;
use std::collections::HashMap;
use std::f32;
use std::future::Future;
use std::iter::*;
use std::marker::PhantomData;
use rand_distr::{Distribution, Gamma};


#[derive(Debug,Clone,Copy)]
pub struct PUCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
    pub pi: f32,
}

#[derive(Debug,Clone)]
pub struct PUCTNodeInfo<G: game::Feature> {
    pub count: f32,
    pub moves: HashMap<G::Move, PUCTMoveInfo>,
}
/**
 * The game state evaluator
 */
pub trait Evaluator<G: game::Feature>:
    Fn(G::Player, &G) -> (Array<f32, G::ActionDim>, f32)
{
}
impl<G: game::Feature, F: Fn(G::Player, &G) -> (Array<f32, G::ActionDim>, f32)> Evaluator<G>
    for F
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
        }
    }
}

pub struct BasePUCTPolicy_<G: game::Feature> {
    pub color: G::Player,
    pub s: PUCTSettings,
    pub tree: HashMap<G, PUCTNodeInfo<G>>,
}

impl<G> BaseMCTSPolicy<G> for BasePUCTPolicy_<G>
where
    G: game::Feature + super::MCTSGame,
{
    type NodeInfo = PUCTNodeInfo<G>;
    type PlayoutInfo = (Option<HashMap<G::Move, f32>>, f32, G);
    //                 (policy, value, leaf_state).

    fn tree(&self) -> &HashMap<G, Self::NodeInfo> {
        &self.tree
    }

    fn tree_mut(&mut self) -> &mut HashMap<G, Self::NodeInfo> {
        &mut self.tree
    }

    fn select_move(&self, board: &G, exploration: bool) -> G::Move {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board).unwrap();
        let N = node_info.count;

        if exploration {
            let moves_scores = moves.map(|action| {
                let v = node_info.moves.get(&action).unwrap();
                let pb_c = ((N + self.s.C_BASE + 1.)/self.s.C_BASE).ln() + self.s.C_INIT;
                let value = v.Q + pb_c * v.pi * (N.sqrt() / (v.N_a + 1.));
                (value, action)
            });
            moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        } else {
            let moves_scores = moves.map(|action| {
                let v = node_info.moves.get(&action).unwrap();
                let value = v.N_a;
                (value, action)
            });
            moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        let moves = HashMap::from_iter(board.possible_moves().map(|m| {
            let mut b_scratch = board.clone();
            b_scratch.play(&m);
            (
                m,
                PUCTMoveInfo {
                    Q: 0.5,
                    N_a: 0.,
                    pi: 1.,
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
            if self.tree.len() == 1 {
                log::info!("PUCT: adding noise.");
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
            for (m, info) in self.tree.get_mut(&board).unwrap().moves.iter_mut() {
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


            let mut node = self.tree.get_mut(&state).unwrap();
            node.count += 1.;
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (value - (*v).Q) / (*v).N_a;
        }
    }
}
/**
 * PUCT
 */
pub struct PUCTPolicy_<G, F>
where
    G: game::Feature,
    F: Evaluator<G>,
{
    pub b: BasePUCTPolicy_<G>,
    pub evaluate: F,
}

impl<G, F> BaseMCTSPolicy<G> for PUCTPolicy_<G, F>
where
    G: game::Feature + super::MCTSGame,
    F: Evaluator<G>,
{
    type NodeInfo = <BasePUCTPolicy_<G> as BaseMCTSPolicy<G>>::NodeInfo;
    type PlayoutInfo = <BasePUCTPolicy_<G> as BaseMCTSPolicy<G>>::PlayoutInfo;

    fn tree(&self) -> &HashMap<G, Self::NodeInfo> {
        self.b.tree()
    }

    fn tree_mut(&mut self) -> &mut HashMap<G, Self::NodeInfo> {
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


impl<G, F> MCTSPolicy<G> for PUCTPolicy_<G, F>
where
    G: game::Feature + super::MCTSGame,
    F: Evaluator<G>,
{
    fn simulate(&self, board: &G) -> <Self as BaseMCTSPolicy<G>>::PlayoutInfo {
        if !board.is_finished() {
            // NN predicts a good policy for current player + expectation of winning from this state.
            let (policy, value) = (self.evaluate)(board.turn(), board);
            let policy = board.feature_to_moves(&policy);
            (Some(policy), value, board.clone())
        } else {
            (None, 0., board.clone())
        }
    }
}

use std::fmt;

/**
 *  POLICY BUILDERS - ASYNC
 */
pub type PUCTPolicy<G, F> = WithMCTSPolicy<G, PUCTPolicy_<G, F>>;

pub struct PUCT<G, F>
where
    G: game::Feature,
    F: (Fn(G::Player, &G) -> (Array<f32, G::ActionDim>, f32)),
{
    pub s: PUCTSettings,
    pub N_PLAYOUTS: usize,
    pub evaluate: F,
    pub _g: PhantomData<fn() -> G>,
}

impl<G, F> fmt::Display for PUCT<G, F>
where
    G: game::Feature,
    F: (Fn(G::Player, &G) -> (Array<f32, G::ActionDim>, f32)),
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BATCHED PUCT")?;
        writeln!(f, "||{:?}", self.s)
    }
}

impl<G, F> MultiplayerPolicyBuilder<G> for PUCT<G, F>
where
    G: game::Feature + super::MCTSGame,
    F: (Fn(G::Player, &G) -> (Array<f32, G::ActionDim>, f32)),
    F: Clone
{
    type P = PUCTPolicy<G, F>;

    fn create(&self, color: G::Player) -> PUCTPolicy<G, F> {
        WithMCTSPolicy::new(
            PUCTPolicy_::<G, F> {
                b: BasePUCTPolicy_::<G> {
                    color,
                    s: self.s,
                    tree: HashMap::new(),
                },
                evaluate: self.evaluate.clone(),
            },
            self.N_PLAYOUTS,
        )
    }
}
