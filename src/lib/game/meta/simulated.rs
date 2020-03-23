use crate::game::*;

use std::marker::PhantomData;
use std::future::Future;
use ndarray::{Array, Dimension};

pub struct NetworkOutput<H: Dimension> {
    reward: f32,
    hidden_state: Array<f32, H>
}

pub trait AsyncNetworkOutput<H>: Future<Output=NetworkOutput<H>> 
where
    H: Dimension 
{}
impl<H,I> AsyncNetworkOutput<H> for I 
where
    H: Dimension,
    I: Future<Output=NetworkOutput<H>>
{}

pub trait AsyncStateEvaluator<G,O,H>: Fn(Array<f32, G::StateDim>) -> O
where 
    G: Feature,
    H: Dimension,
    O: AsyncNetworkOutput<H>
{}
impl<G,O,H,I> AsyncStateEvaluator<G,O,H> for I
where 
    G: Feature,
    H: Dimension,
    O: AsyncNetworkOutput<H>,
    I: Fn(Array<f32, G::StateDim>) -> O
{}

pub trait AsyncDynamicsEvaluator<G,O,H>: Fn(Array<f32, H>, Array<f32, G::ActionDim>) -> O
where
    G: Feature,
    H: Dimension,
    O: AsyncNetworkOutput<H>
{}
impl<G,O,H,I> AsyncDynamicsEvaluator<G,O,H> for I
where
    G: Feature,
    H: Dimension,
    O: AsyncNetworkOutput<H>,
    I: Fn(Array<f32, H>, Array<f32, G::ActionDim>) -> O
{}

#[derive(Debug,Clone,Copy)]
pub struct Simulated<G, H, O, SE, SD>
where
    G: Feature,
    H: Dimension,
    O: AsyncNetworkOutput<H>,
    SE: AsyncStateEvaluator<G,O,H>,
    SD: AsyncDynamicsEvaluator<G,O,H>
{
    _g: PhantomData<G>,
    hidden_state: Array<f32, H>,
    total_reward: f32,
    state_evaluator: SE,
    dynamics_evaluator: SD,
}

type PossibleMovesIterator<'a, G: Feature> = impl Iterator<Item=<G::ActionDim as Dimension>::SliceArg> + 'a;

use async_trait::async_trait;

impl<G, H, O, SE, SD> Base for Simulated<G, H, O, SE, SD> 
where
    G: Feature,
    H: Dimension,
    O: AsyncNetworkOutput<H>,
    SE: AsyncStateEvaluator<G,O,H>,
    SD: AsyncDynamicsEvaluator<G,O,H>
{
    type Move = <G::ActionDim as Dimension>::SliceArg;
    type MoveIterator<'a> = PossibleMovesIterator<'a, G>;

    fn hash(&self) -> usize {
        0
    }

    fn possible_moves<'a>(&'a self) -> Self::MoveIterator<'a> {
        let dimensions = G::action_dimension().size();
        
    }
}

