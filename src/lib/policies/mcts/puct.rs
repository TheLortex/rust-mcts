use crate::game;
use crate::policies::mcts::{BaseMCTSPolicy, MCTSNode, WithMCTSPolicy};
use crate::policies::MultiplayerPolicyBuilder;

use float_ord::FloatOrd;
use ndarray::Array;
use rand_distr::{Distribution, Gamma};
use std::collections::HashMap;
use std::f32;
use std::iter::*;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct PUCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
    pub pi: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PUCTNodeInfo {
    pub count: f32,
}
/**
 * The game state evaluator
 */
pub trait Evaluator<G: game::Feature>:
    Fn(G::Player, &G) -> (Array<f32, G::ActionDim>, f32)
{
}
impl<G: game::Feature, F: Fn(G::Player, &G) -> (Array<f32, G::ActionDim>, f32)> Evaluator<G> for F {}
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

#[derive(Clone)]
pub struct PUCTPolicy_<G: game::Feature, F>
where
    F: Evaluator<G>,
{
    pub color: G::Player,
    pub s: PUCTSettings,
    pub evaluate: F,
}

impl<G, F> BaseMCTSPolicy<G> for PUCTPolicy_<G, F>
where
    G: game::Feature + super::MCTSGame,
    F: Evaluator<G>,
{
    type NodeInfo = PUCTNodeInfo;
    type MoveInfo = PUCTMoveInfo;
    type PlayoutInfo = (Option<HashMap<G::Move, f32>>, f32, G::Player);

    fn get_value(
        &self,
        board: &G,
        action: &G::Move,
        node_info: &Self::NodeInfo,
        move_info: &Self::MoveInfo,
        exploration: bool,
    ) -> f32 {
        if exploration {
            let N = node_info.count;
            let v = move_info;
            let pb_c = ((N + self.s.C_BASE + 1.) / self.s.C_BASE).ln() + self.s.C_INIT;
            v.Q + pb_c * v.pi * (N.sqrt() / (v.N_a + 1.))
        } else {
            move_info.N_a
        }
    }

    fn default_move(&self, board: &G, action: &G::Move) -> Self::MoveInfo {
        PUCTMoveInfo {
            Q: 0.5,
            N_a: 0.,
            pi: 1.,
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        PUCTNodeInfo { count: 0. }
    }

    fn backpropagate_new_node(
        &self,
        node: &mut MCTSNode<G, Self>,
        history: &[G::Move],
        (policy, value, pov): &Self::PlayoutInfo,
    ) {
        if let Some(mut policy) = policy.clone() {
            // save probabilities of newly created node.
            
            if history.is_empty() {
                log::info!("PUCT: adding noise.");
                // add dirichlet noise on the root node.
                let frac = self.s.ROOT_EXPLORATION_FRACTION;
                let gamma = Gamma::new(self.s.ROOT_DIRICHLET_ALPHA, 1.0).unwrap();
                for (_, val) in policy.iter_mut() {
                    let noise = gamma.sample(&mut rand::thread_rng());
                    *val = frac * (*val) + (1. - frac) * noise;
                }
            }
            let z: f32 = node.moves.keys().map(|m| policy.get(&m).unwrap()).sum();
            let z = if z == 0. { 1. } else { z };
            for (m, info) in node.moves.iter_mut() {
                info.pi = policy.get(&m).unwrap() / z;
            }
        };
    }

    fn backpropagate(
        &self,
        _index: usize,
        info: &mut MCTSNode<G, Self>,
        action: &G::Move,
        history: &[G::Move],
        (policy, value, pov): &Self::PlayoutInfo,
    ) {
        let value = if info.state.turn() == *pov {
            *value
        } else {
            1. - *value
        };

        info.node.count += 1.;

        let mut v = info.moves.get_mut(action).unwrap();
        (*v).N_a += 1.;
        (*v).Q += (value - (*v).Q) / (*v).N_a;
    }

    fn simulate(&self, board: &G) -> Self::PlayoutInfo {
        if !board.is_finished() {
            // NN predicts a good policy for current player + expectation of winning from this state.
            let (policy, value) = (self.evaluate)(board.turn(), board);
            let policy = board.feature_to_moves(&policy);
            (Some(policy), value, board.turn())
        } else {
            (None, 0., board.turn())
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
                color,
                s: self.s,
                evaluate: self.evaluate.clone(),
            },
            self.N_PLAYOUTS,
        )
    }
}
