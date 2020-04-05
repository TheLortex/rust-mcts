// CODE TAKEN AND MODIFIED FROM https://raw.githubusercontent.com/MrRobb/gym-rs/master/src/lib.rs under MIT License

#[macro_use]
use failure;
use ndarray;
use rand;
use pyo3::prelude::*;

use failure::Fail;
use serde::{Serialize, Deserialize};
use rand::Rng;

type DiscreteType = usize;
type VectorType<T> = ndarray::Array1<T>;
pub type Action = SpaceData;
pub type Observation = SpaceData;
pub type Reward = f64;

#[derive(Debug, Fail)]
pub enum GymError {
    #[fail(display = "Invalid action")]
    InvalidAction,
    #[fail(display = "Invalid conversion")]
    InvalidConversion,
    #[fail(display = "Wrong type")]
    WrongType,
    #[fail(display = "Unable to parse step result")]
    WrongStepResult,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct State {
    pub observation: SpaceData,
    pub reward: f64,
    pub is_done: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum SpaceTemplate {
    DISCRETE {
        n: DiscreteType,
    },
    BOX {
        high: Vec<f64>,
        low: Vec<f64>,
        shape: Vec<usize>,
    },
    TUPLE {
        spaces: Vec<SpaceTemplate>,
    },
}

#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub enum SpaceData {
    DISCRETE(DiscreteType),
    BOX(VectorType<f64>),
    TUPLE(VectorType<SpaceData>),
}

pub struct Environment {
    env: PyObject,
    observation_space: SpaceTemplate,
    action_space: SpaceTemplate,
}

pub struct GymClient {
    gym: Py<PyModule>,
    version: String,
}

impl SpaceData {
    pub fn get_discrete(self) -> Result<DiscreteType, GymError> {
        match self {
            SpaceData::DISCRETE(n) => Ok(n),
            _ => Err(GymError::WrongType),
        }
    }

    pub fn get_box(self) -> Result<VectorType<f64>, GymError> {
        match self {
            SpaceData::BOX(v) => Ok(v),
            _ => Err(GymError::WrongType),
        }
    }

    pub fn get_tuple(self) -> Result<VectorType<SpaceData>, GymError> {
        match self {
            SpaceData::TUPLE(s) => Ok(s),
            _ => Err(GymError::WrongType),
        }
    }

    pub fn into_pyo(self) -> Result<PyObject, GymError> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match self {
            SpaceData::DISCRETE(n) => n.into_py(py),
            SpaceData::BOX(v) => v.to_vec().into_py(py),
            SpaceData::TUPLE(spaces) => {
                let vpyo = spaces
                    .to_vec()
                    .into_iter()
                    .map(|s| s.into_pyo().expect("Unable to parse tuple"))
                    .collect::<Vec<_>>();
                vpyo.into_py(py)
            }
        })
    }
}

impl SpaceTemplate {
    fn extract_data(&self, pyo: PyObject) -> Result<SpaceData, GymError> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        match self {
            SpaceTemplate::DISCRETE { .. } => {
                let n = pyo
                    .extract::<DiscreteType>(py)
                    .map_err(|_| GymError::InvalidConversion)?;
                Ok(SpaceData::DISCRETE(n))
            }
            SpaceTemplate::BOX { .. } => {
                let v = pyo
                    .call_method(py, "flatten", (), None)
                    .map_err(|_| GymError::InvalidConversion)?
                    .extract::<Vec<f64>>(py)
                    .map_err(|_| GymError::InvalidConversion)?;
                Ok(SpaceData::BOX(v.into()))
            }
            SpaceTemplate::TUPLE { .. } => {
                unimplemented!("Never used for now.")
                /*
                let mut tuple = vec![];
                let mut i = 0;
                let mut item = pyo.cast_as::<pyo3::types::PyTuple>(py).unwrap().get_item(i);
                while item.is_ok() {
                    let pyo_item = self.extract_data(item.unwrap())?;
                    tuple.push(pyo_item);
                    i += 1;
                    item = pyo.get_item(py, i);
                }
                Ok(SpaceData::TUPLE(tuple.into()))*/
            }
        }
    }

    fn extract_template(pyo: PyObject) -> SpaceTemplate {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let class = pyo
            .getattr(py, "__class__")
            .expect("Unable to extract __class__ (this should never happen)");

        let name = class
            .getattr(py, "__name__")
            .expect("Unable to extract __name__ (this should never happen)")
            .extract::<String>(py)
            .expect("Unable to extract __name__ (this should never happen)");

        match name.as_ref() {
            "Discrete" => {
                let n = pyo
                    .getattr(py, "n")
                    .expect("Unable to get attribute 'n'")
                    .extract::<usize>(py)
                    .expect("Unable to convert 'n' to usize");
                SpaceTemplate::DISCRETE { n }
            }
            "Box" => {
                let high = pyo
                    .getattr(py, "high")
                    .expect("Unable to get attribute 'high'")
                    .call_method(py, "flatten", (), None)
                    .expect("Unable to call 'flatten'")
                    .extract::<Vec<f64>>(py)
                    .expect("Unable to convert 'high' to Vec<f64>");

                let low = pyo
                    .getattr(py, "low")
                    .expect("Unable to get attribute 'low'")
                    .call_method(py, "flatten", (), None)
                    .expect("Unable to call 'flatten'")
                    .extract::<Vec<f64>>(py)
                    .expect("Unable to convert 'low' to Vec<f64>");

                let shape = pyo
                    .getattr(py, "shape")
                    .expect("Unable to get attribute 'shape'")
                    .extract::<Vec<usize>>(py)
                    .expect("Unable to convert 'shape' to Vec<f64>");

                debug_assert_eq!(high.len(), low.len());
                debug_assert_eq!(low.len(), shape.iter().product::<usize>());
                high.iter()
                    .zip(low.iter())
                    .for_each(|(h, l)| debug_assert!(h > l));

                SpaceTemplate::BOX { high, low, shape }
            }
            "Tuple" => {
                unimplemented!("Tuple is not supported.");
                /*
                let mut i = 0;
                let mut tuple = vec![];
                let mut item = pyo.get_item(py, i);

                while item.is_ok() {
                    let pyo_item = item.unwrap();
                    let space = SpaceTemplate::extract_template(pyo_item);
                    tuple.push(space);
                    i += 1;
                    item = pyo.get_item(py, i);
                }

                SpaceTemplate::TUPLE { spaces: tuple }*/
            }
            _ => unreachable!(),
        }
    }

    pub fn sample(&self) -> SpaceData {
        let mut rng = rand::thread_rng();
        match self {
            SpaceTemplate::DISCRETE { n } => SpaceData::DISCRETE(rng.gen_range(0, n)),
            SpaceTemplate::BOX { high, low, shape } => {
                let dimensions = shape.len();
                let mut v = vec![];
                for d in 0..dimensions {
                    for _ in 0..shape[d] {
                        v.push(rng.gen_range(low[d], high[d]));
                    }
                }
                SpaceData::BOX(v.into())
            }
            SpaceTemplate::TUPLE { spaces } => {
                let mut tuple = vec![];
                for space in spaces {
                    let sample = space.sample();
                    tuple.push(sample);
                }
                SpaceData::TUPLE(tuple.into())
            }
        }
    }
}

impl Environment {
    pub fn seed(&self, seed: u64) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.env
            .call_method(py, "seed", (seed,), None)
            .expect("Unable to call 'seed'");
    }

    pub fn reset(&self) -> Result<SpaceData, GymError> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let result = self
            .env
            .call_method(py, "reset", (), None)
            .expect("Unable to call 'reset'");
        self.observation_space.extract_data(result)
    }

    pub fn render(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.env
            .call_method(py, "render", (), None)
            .expect("Unable to call 'render'");
    }

    pub fn step(&self, action: &Action) -> Result<State, GymError> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let result = match action {
            Action::DISCRETE(n) => self
                .env
                .call_method(py, "step", (*n,), None)
                .map_err(|_| GymError::InvalidAction)?,
            Action::BOX(v) => {
                let vv = v.to_vec();
                self.env
                    .call_method(py, "step", (vv,), None)
                    .map_err(|_| GymError::InvalidAction)?
            }
            Action::TUPLE(spaces) => {
                unimplemented!("Tuple is not supported.");
                /*
                let vpyo = spaces
                    .to_vec()
                    .into_iter()
                    .map(|s| s.into_pyo().unwrap())
                    .collect::<Vec<_>>();
                let tpyo = PyTuple::new(py, &vpyo);
                self.env
                    .call_method(py, "step", (tpyo,), None)
                    .map_err(|_| GymError::InvalidAction)?*/
            }
        };

        let s = State {
            observation: self.observation_space.extract_data(
                result
                    .cast_as::<pyo3::types::PyTuple>(py)
                    .map(|x| x.get_item(0).to_object(py))
                    .map_err(|_| GymError::WrongStepResult)?,
            )?,
            reward: result
                .cast_as::<pyo3::types::PyTuple>(py)
                .map(|x| x.get_item(1))
                .map_err(|_| GymError::WrongStepResult)?
                .extract()
                .map_err(|_| GymError::WrongStepResult)?,
            is_done: result
                .cast_as::<pyo3::types::PyTuple>(py)
                .map(|x| x.get_item(2))
                .map_err(|_| GymError::WrongStepResult)?
                .extract()
                .map_err(|_| GymError::WrongStepResult)?,
        };

        Ok(s)
    }

    pub fn close(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let _ = self
            .env
            .call_method(py, "close", (), None)
            .expect("Unable to call 'close'");
    }

    /// Returns the number of allowed actions for this environment.
    pub fn action_space(&self) -> &SpaceTemplate {
        &self.action_space
    }

    /// Returns the shape of the observation tensors.
    pub fn observation_space(&self) -> &SpaceTemplate {
        &self.observation_space
    }
}

impl Default for GymClient {
    fn default() -> Self {
        log::debug!("Getting Python.");
        pyo3::prepare_freethreaded_python();
        // Get python
        let gil = Python::acquire_gil();
        let py = gil.python();

        log::debug!("Got Python.");
        // Set argv[0] -> otherwise render() fails
        let sys = py.import("sys").expect("Error: import sys");

        match sys.get("argv") {
            Result::Ok(argv) => {
                argv.call_method("append", ("",), None)
                    .expect("Error: sys.argv.append('')");
            }
            Result::Err(_) => {}
        };

        log::debug!("Importing Gym.");
        // Import gym
        let gym = py.import("gym").expect("Error: import gym");
        log::debug!("Imported Gym.");
        let version: String = gym
            .get("__version__")
            .expect("Unable to call gym.__version__")
            .extract()
            .expect("Unable to call gym.__version__");

        log::debug!("Gym version: {}", version);
        GymClient { gym: Py::from(gym), version }
    }
}

impl GymClient {
    pub fn make(&self, env_id: &str) -> Environment {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let env = self
            .gym
            .as_ref(py)
            .call("make", (env_id,), None)
            .expect("Unable to call 'make'")
            .to_object(py);

        Environment {
            observation_space: SpaceTemplate::extract_template(
                env.getattr(py, "observation_space")
                    .expect("Unable to get attribute 'observation_space'")
                    .to_object(py),
            ),
            action_space: SpaceTemplate::extract_template(
                env.getattr(py, "action_space")
                    .expect("Unable to get attribute 'action_space'")
                    .to_object(py),
            ),
            env,
        }
    }

    pub fn version(&self) -> &str {
        self.version.as_str()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_gym_client() {
        let _client = GymClient::default();
    }

    #[test]
    fn test_make() {
        let client = GymClient::default();
        client.make("CartPole-v1");
    }

    #[test]
    fn test_seed() {
        let client = GymClient::default();
        let env = client.make("FrozenLake-v0");
        env.seed(1002);
        let obs = env.reset().unwrap();
        assert_eq!(0, obs.get_discrete().unwrap());
        let action = SpaceData::DISCRETE(1);
        let state = env.step(&action).unwrap();
        assert_eq!(4, state.observation.get_discrete().unwrap());
    }

    #[test]
    fn test_reset() {
        let client = GymClient::default();
        let env = client.make("CartPole-v1");
        env.reset().unwrap();
    }

    #[test]
    fn test_box_observation_3d() {
        let client = GymClient::default();
        let env = client.make("VideoPinball-v0");
        env.reset().unwrap();
        env.step(&env.action_space().sample()).unwrap();
    }

    #[test]
    fn test_step() {
        let client = GymClient::default();
        let env = client.make("CartPole-v1");
        env.reset().unwrap();
        let action = env.action_space().sample();
        env.step(&action).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_invalid_action() {
        let client = GymClient::default();
        let env = client.make("CartPole-v1");
        env.reset().unwrap();
        let action = Action::DISCRETE(500); // invalid action
        env.step(&action).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_wrong_type() {
        let client = GymClient::default();
        let env = client.make("CartPole-v1");
        env.reset().unwrap();
        let _ = env.action_space().sample().get_box().unwrap();
    }

    #[test]
    fn test_box_action() {
        let client = GymClient::default();
        let env = client.make("BipedalWalker-v3");
        env.reset().unwrap();
        let action = env.action_space().sample();
        env.step(&action).unwrap();
    }

    #[test]
    fn test_tuple_template() {
        let client = GymClient::default();
        let _ = client.make("Blackjack-v0");
    }

    #[test]
    fn test_tuple_obs() {
        let client = GymClient::default();
        let env = client.make("Blackjack-v0");
        env.reset().unwrap();
        let action = env.action_space().sample();
        env.step(&action).unwrap();
    }

    #[test]
    fn test_tuple_action() {
        let client = GymClient::default();
        let env = client.make("RepeatCopy-v0");
        env.reset().unwrap();
        let action = env.action_space().sample();
        env.step(&action).unwrap();
    }

    #[test]
    fn test_gym_version() {
        let client = GymClient::default();
        assert!(!client.version().is_empty())
    }

    #[test]
    fn test_render() {
        let client = GymClient::default();
        let env = client.make("FrozenLake-v0");
        env.reset().unwrap();
        let action = env.action_space().sample();
        env.step(&action).unwrap();
        env.render();
    }

    #[test]
    fn test_close() {
        let client = GymClient::default();
        let env = client.make("FrozenLake-v0");
        env.close();
    }
}
