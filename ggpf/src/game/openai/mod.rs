use crate::game::*;

use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;
use std::sync::Mutex;
//pub mod gym;

/*
#[derive(Clone)]
pub struct Gym {
    pub env: Arc<Mutex<gym::Environment>>,
    possible_moves: Vec<usize>,
    is_done: bool,
    current_state: Array<f32, Ix3>,
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

impl Gym {
    pub fn new(gym: &gym::GymClient) -> Self {
        //let gym = gym::GymClient::default();
        let env = gym.make("CartPole-v1");

        let possible_moves = match env.action_space() {
            gym::SpaceTemplate::DISCRETE { n } => (0..*n).collect::<Vec<_>>(),
            x => panic!("Unsupported action space. {:?}", x),
        };

        let init_state = env.reset().unwrap();

        let state_dimension = match env.observation_space() {
            gym::SpaceTemplate::BOX { high, low, shape } => convert_to_3D(shape),
            _ => panic!("..."),
        };

        let obs_state = init_state
            .get_box()
            .unwrap()
            .mapv(|x| x as f32)
            .into_shape(state_dimension)
            .expect("Unable to reshape observation.");

        Self {
            env: Arc::new(Mutex::new(env)),
            possible_moves,
            is_done: false,
            current_state: obs_state,
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
        let env = self.env.lock().unwrap();
        let next_state = env.step(&gym::SpaceData::DISCRETE(*action)).unwrap();
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
        let possible_moves = self.possible_moves.clone();
        let action_dimension = Ix1(possible_moves.len());

        let env = self.env.lock().unwrap();
        let state_dimension = match env.observation_space() {
            gym::SpaceTemplate::BOX { high, low, shape } => convert_to_3D(shape),
            _ => panic!("..."),
        };
        (possible_moves, state_dimension, action_dimension)
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
/*
impl Game for Gym {
    type Player = ();
    type Move = ();
    type GameHash = ();

    fn new(_: Self::Player) -> Gym {
        let gym = gym::GymClient::default();
        let env = gym.make("CartPole-v1");
        Gym {
            env
        }
    }

    fn players() -> Vec<()> {
        vec![()]
    }

    fn score(&self, _: Self::Player) -> f32 {panic!("")}

    fn play(&mut self, m: &Self::Move) {panic!("")}

    fn turn(&self) -> Self::Player {panic!("")}

    fn hash(&self) -> Self::GameHash {panic!("")}

    fn possible_moves(&self) -> &Vec<Self::Move> {panic!("")}

    fn winner(&self) -> Option<Self::Player> {panic!("")}

    fn pass(&mut self) {panic!("")}
}*/
*/