use crate::game::*;

use ndarray::s;
use std::fmt;
use std::iter::FromIterator;

use ggpf_gym::*;

#[derive(Clone)]
/// OpenAI Gym game instance.
/// 
/// Each instance is connected to a remote runner. This is because of Rust limitations
/// somehow it's not possible to have both `tensorflow` and `pyo3` in the same crate..
pub struct Gym {
    env: GymRunnerClient,
    game: String,
    possible_moves: Vec<usize>,
    is_done: bool,
    current_state: Array<f32, Ix3>,
    features: (Vec<usize>, Ix3, Ix1),
}

impl fmt::Debug for Gym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TODO")
    }
}

fn convert_to_3D(shape: &[usize]) -> Ix3 {
    if shape.len() > 3 {
        panic!("Observation dimension is too big.")
    };
    let mut dim = Ix3(1, 1, 1);
    for i in 0..shape.len() {
        dim[3 - shape.len() + i] = shape[i]
    }
    dim
}

fn interpolate(image: &Array<f32, Ix3>, x_t: usize, y_t: usize) -> Array<f32, Ix3> {
    let x_o = image.shape()[0] as f32;
    let y_o = image.shape()[1] as f32;

    let mut result = Array::zeros([x_t, y_t, 3]);
    for x in 0..x_t {
        for y in 0..y_t {
            let x_r = (x as f32 / x_t as f32) * x_o;
            let x_0 = x_r.floor();
            let x_1 = x_r.ceil();
            let y_r = (y as f32 / y_t as f32) * y_o;
            let y_0 = y_r.floor();
            let y_1 = y_r.ceil();

            let f_0_0 = image.slice(s![x_0 as usize, y_0 as usize, ..]);
            let f_0_1 = image.slice(s![x_0 as usize, y_1 as usize, ..]);
            let f_1_0 = image.slice(s![x_1 as usize, y_0 as usize, ..]);
            let f_1_1 = image.slice(s![x_1 as usize, y_1 as usize, ..]);

            let dx = x_r - x_0;
            let dy = y_r - y_0;

            let delta_f_x = &f_1_0 - &f_0_0;
            let delta_f_y = &f_0_1 - &f_0_0;
            let delta_f_xy = &f_1_1 + &f_0_0 - f_1_0 - f_0_1;

            result
                .slice_mut(s![x, y, ..])
                .assign(&(dx * delta_f_x + dy * delta_f_y + dx * dy * delta_f_xy + f_0_0));
        }
    }
    result
}

use tarpc::context;

impl Gym {
    /// Given a connected client, build a game based on Gym.
    pub async fn new(mut env: GymRunnerClient, game: String) -> Self {
        let possible_moves = match env.action_space(context::current()).await.unwrap() {
            gym::SpaceTemplate::DISCRETE { n } => (0..n).collect::<Vec<_>>(),
            x => panic!("Unsupported action space. {:?}", x),
        };

        let init_state = env.reset(context::current()).await.unwrap();

        let state_dimension = match env.observation_space(context::current()).await.unwrap() {
            gym::SpaceTemplate::BOX { shape, .. } => convert_to_3D(&shape),
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
            features: (possible_moves, state_dimension, action_dimension),
            game,
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
        let (pm, st, ac) = self.features.clone();
        if self.game == "Breakout-v0" {
            (pm, ndarray::Dim([96, 96, 3]), ac)
        } else {
            (pm, st, ac)
        }
    }

    fn state_dimension(descr: &Self::Descriptor) -> Self::StateDim {
        descr.1
    }

    fn action_dimension(descr: &Self::Descriptor) -> Self::ActionDim {
        descr.2
    }

    fn state_to_feature(&self, _pov: Self::Player) -> Array<f32, Self::StateDim> {
        let res = self.current_state.clone();

        if self.game == "Breakout-v0" {
            interpolate(&res, 96, 96)
        } else {
            res
        }
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
        _descr: &Self::Descriptor,
        features: &Array<f32, Self::ActionDim>,
    ) -> HashMap<Self::Move, f32> {
        HashMap::from_iter(features.iter().cloned().enumerate())
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        HashMap::from_iter(features.iter().cloned().enumerate())
    }
}

#[derive(Clone, Debug)]
/// Builder for Gym games.
pub struct GymBuilder {
    /// Executor remote address.
    pub address: String,
    /// Gym game name.
    pub game_name: String,
    /// Whether the game should be rendered.
    pub render: bool,
}

use tarpc::client;

#[async_trait]
impl SingleplayerGameBuilder for GymBuilder {
    type G = Gym;

    async fn create(&self) -> Gym {
        let conn = tarpc::serde_transport::tcp::connect(&self.address, BinCodec::default());
        let conn = conn.await.unwrap();

        let mut runner = GymRunnerClient::new(client::Config::default(), conn)
            .spawn()
            .unwrap();
        runner
            .init(context::current(), self.game_name.clone(), self.render)
            .await
            .unwrap();
        runner.reset(context::current()).await.unwrap();

        Gym::new(runner, self.game_name.clone()).await
    }
}
