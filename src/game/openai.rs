mod gym;

use super::Game;

use std::fmt;

#[derive(Clone)]
struct Gym<'a> {
    env: gym::Environment<'a>,
    score: f32
}

impl<'a> fmt::Debug for Gym<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>)-> fmt::Result {
        write!(f, "TODO")
    }
}

impl<'a> Game for Gym<'a> {
    type Player = ();
    type Move = gym::Action;
    type GameHash = ();
    type Settings = &'a gym::GymClient;
    
    fn new(_: Self::Player, client: Self::Settings) -> Gym<'a> {
        let env = client.make("MsPacman-v0");
        Gym {
            env,
            score: 0.
        }
    }

    fn players() -> Vec<()> {
        vec![()]
    }

    fn score(&self, _: Self::Player) -> f32 {
        self.score
    }

    fn play(&mut self, m: &Self::Move) {
        self.env.step(m);
    }

    fn turn(&self) -> Self::Player {

    }

    fn hash(&self) -> Self::GameHash {panic!("")}

    fn possible_moves(&self) -> &Vec<Self::Move> {panic!("")}

    fn winner(&self) -> Option<Self::Player> {panic!("")}

    fn pass(&mut self) {panic!("")}
}