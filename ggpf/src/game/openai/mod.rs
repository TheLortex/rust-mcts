use crate::game::*;

use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;
use std::sync::Mutex;

use ggpf_gym::*;


#[derive(Clone)]
pub struct Gym {
    env: GymRunnerClient,
    possible_moves: Vec<usize>,
    is_done: bool,
    current_state: Array<f32, Ix3>,
    features: (Vec<usize>, Ix3, Ix1),
}

unsafe impl Send for Gym {}
unsafe impl Sync for Gym {}

impl fmt::Debug for Gym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TODO")
    }
}

fn convert_to_3D(shape: &Vec<usize>) -> Ix3 {
    if shape.len() > 3 {
        panic!("Observation dimension is too big.")
    };
    let mut dim = Ix3(1, 1, 1);
    for i in 0..shape.len() {
        dim[3 - shape.len() + i] = shape[i]
    }
    dim
}

use tarpc::context;

impl Gym {
    pub async fn new(mut env: GymRunnerClient) -> Self {

        let possible_moves = match env.action_space(context::current()).await.unwrap() {
            gym::SpaceTemplate::DISCRETE { n } => (0..n).collect::<Vec<_>>(),
            x => panic!("Unsupported action space. {:?}", x),
        };

        let init_state = env.reset(context::current()).await.unwrap();

        let state_dimension = match env.observation_space(context::current()).await.unwrap() {
            gym::SpaceTemplate::BOX { high, low, shape } => convert_to_3D(&shape),
            _ => panic!("..."),
        };

        let obs_state = init_state
            .get_box()
            .unwrap()
            .mapv(|x| x as f32)
            .into_shape(state_dimension)
            .expect("Unable to reshape observation.");

        let action_dimension = Ix1(possible_moves.len());

        Self {
            env,
            possible_moves: possible_moves.clone(),
            is_done: false,
            current_state: obs_state,
            features: (possible_moves, state_dimension, action_dimension)
        }
    }
}

impl Base for Gym {
    type Move = usize;

    fn possible_moves(&self) -> Vec<Self::Move> {
        self.possible_moves.clone()
    }

    fn is_finished(&self) -> bool {
        self.is_done
    }
}

use async_trait::async_trait;

#[async_trait]
impl Playable for Gym {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    async fn play(&mut self, action: &usize) -> f32 {
        //let env = self.env.lock().unwrap();
        let next_state = self.env.play(context::current(), *action).await.unwrap();
        self.is_done = next_state.is_done;
        next_state.reward as f32
    }
}

use ndarray::{Ix1, Ix3};

impl Singleplayer for Gym {}

impl Features for Gym {
    type StateDim = Ix3;
    type ActionDim = Ix1;

    type Descriptor = (Vec<usize>, Ix3, Ix1);

    fn get_features(&self) -> Self::Descriptor {
        self.features.clone()
    }

    fn state_dimension(descr: &Self::Descriptor) -> Self::StateDim {
        descr.1
    }

    fn action_dimension(descr: &Self::Descriptor) -> Self::ActionDim {
        descr.2
    }

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim> {
        self.current_state.clone()
    }

    fn all_possible_moves(descr: &Self::Descriptor) -> Vec<Self::Move> {
        descr.0.clone()
    }

    fn moves_to_feature(
        descr: &Self::Descriptor,
        moves: &HashMap<Self::Move, f32>,
    ) -> Array<f32, Self::ActionDim> {
        let mut res = Array::zeros(descr.2);
        for (i, v) in moves.iter() {
            res[*i] = *v;
        }

        res
    }

    fn all_feature_to_moves(
        descr: &Self::Descriptor,
        features: &Array<f32, Self::ActionDim>,
    ) -> HashMap<Self::Move, f32> {
        HashMap::from_iter(features.iter().cloned().enumerate())
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        HashMap::from_iter(features.iter().cloned().enumerate())
    }
}

#[derive(Clone, Debug)]
pub struct GymBuilder {
    pub address: String,
    pub game_name: String,
    pub render: bool,
}

use tarpc::{
    client,
};


#[async_trait]
impl SingleplayerGameBuilder<Gym> for GymBuilder {
    async fn create(&self) -> Gym {

        let conn = tarpc::serde_transport::tcp::connect(&self.address, BinCodec::default());
        let conn = conn.await.unwrap();

        let mut runner =
        GymRunnerClient::new(client::Config::default(), conn).spawn().unwrap();
        runner.init(context::current(), self.game_name.clone(), self.render).await.unwrap();
        runner.reset(context::current()).await.unwrap();

        Gym::new(runner).await
    }
}
