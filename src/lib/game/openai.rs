mod gym;

use super::Game;

use std::fmt;

#[derive(Clone)]
struct Gym {
    env: gym::Environment<'_>
}

impl fmt::Debug for Gym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>)-> fmt::Result {
        write!(f, "TODO")
    }
}

impl Clone for Gym {

}

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
}