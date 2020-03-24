use crate::game::*;

use std::marker::PhantomData;
use ndarray::{Array, Dimension};

use std::iter::FromIterator;

pub struct NetworkOutput<H: Dimension> {
    pub reward: f32,
    pub hidden_state: Array<f32, H>
}


pub trait StateEvaluator<G,H>: Fn(Array<f32, G::StateDim>) -> NetworkOutput<H>
where 
    G: Feature,
    H: Dimension,
{}

impl<G,H,I> StateEvaluator<G,H> for I
where 
    G: Feature,
    H: Dimension,
    I: Fn(Array<f32, G::StateDim>) -> NetworkOutput<H>
{}

pub trait DynamicsEvaluator<G,H>: Clone + Fn(Array<f32, H>, Array<f32, G::ActionDim>) -> NetworkOutput<H>
where
    G: Feature,
    H: Dimension,
{}
impl<G,H,I> DynamicsEvaluator<G,H> for I
where
    G: Feature,
    H: Dimension,
    I: Clone + Fn(Array<f32, H>, Array<f32, G::ActionDim>) -> NetworkOutput<H>
{}

pub struct Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H>
{
    _g: PhantomData<G>,
    hidden_state: Array<f32, H>,
    total_reward: f32,
    dynamics_evaluator: DE,
    hidden_dimension: H,
}

impl<G, H, DE> Clone for  Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H> 
{
    fn clone(&self) -> Self {
        Self {
            _g: PhantomData,
            hidden_dimension: self.hidden_dimension.clone(),
            hidden_state: self.hidden_state.clone(),
            dynamics_evaluator: self.dynamics_evaluator.clone(),
            total_reward: self.total_reward
        }
    }
}

impl<G, H, DE> Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H> 
{
    pub fn new(hidden_state: Array<f32, H>, dynamics_evaluator: DE) -> Self {
        let hidden_dimension = hidden_state.raw_dim();
        Simulated {
            _g: PhantomData,
            hidden_state,
            total_reward: 0.,
            hidden_dimension,
            dynamics_evaluator
        }
    }
}



use std::fmt::*;

impl<G, H, DE> Debug for Simulated<G, H, DE>
where
    G: Feature,
    H: Dimension,
    DE: DynamicsEvaluator<G, H>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"Simulated: {:?}\n{:?}", self.total_reward, self.hidden_state)
    }
}

impl<G, H, SD> Base for Simulated<G, H, SD> 
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>
{
    type Move = usize;

    #[allow(clippy::needless_lifetimes)]
    fn possible_moves<'a>(&'a self) -> Vec<Self::Move> {
        (0..G::action_dimension().size()).collect()
    }
}

impl<G, H, SD> Playable for Simulated<G, H, SD> 
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>
{
    fn play(&mut self, action: &Self::Move) {

    }
}

impl<G, H, SD> Game for Simulated<G, H, SD> 
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>
{
    type Player = ();

    fn players() -> Vec<Self::Player> {
        vec![()]
    }

    fn turn(&self) -> Self::Player {}

    fn has_won(&self, _player: Self::Player) -> bool {
        false
    }
}

impl<G, H, SD> Feature for Simulated<G, H, SD> 
where
    G: Feature + 'static,
    H: Dimension,
    SD: DynamicsEvaluator<G, H>
{
    type StateDim = H;
    type ActionDim = G::ActionDim;

    fn state_dimension(&self) -> Self::StateDim {
        self.hidden_dimension.clone()
    }

    fn state_to_feature(&self, _pov: Self::Player) -> Array<f32, Self::StateDim> {
        self.hidden_state.clone()
    }

    fn action_dimension() -> Self::ActionDim {
        G::action_dimension()
    }

    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim> {
        let mut features = ndarray::Array::zeros(Self::action_dimension());

        for (action, proba) in moves.iter() {
            features.as_slice_mut().unwrap()[*action] = *proba;
        }

        features
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        HashMap::from_iter(
            features.iter().enumerate().map(|(i,x)| (i,*x))
        )
    }
}

