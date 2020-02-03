
use std::collections::HashMap;
use std::f64;
use std::iter::*;

use super::game::Game;
use super::policies::{Policy, N_PLAYOUTS};

#[derive(Debug)]
struct MoveInfo {
    wins: f64,
    count: f64,
    wins_AMAF: f64,
    count_AMAF: f64,
}

#[derive(Debug)]
struct NodeInfo<G: Game> {
    count: f64,
    moves: HashMap<G::Move, MoveInfo>,
}

pub struct RAVEPolicy<G: Game> {
    color: G::Player,
    tree: HashMap<G::GameHash, NodeInfo<G>>,
}

impl<G: Game> RAVEPolicy<G> {
    fn simulate(self: &mut RAVEPolicy<G>, board: &G) {
        let mut b = *board; // COPY BOARD
        let history = self.sim_tree(&mut b);
        let (z, history_default) = b.playout_history();
        self.update(history, history_default, z);
    }

    fn update(
        self: &mut RAVEPolicy<G>,
        history: Vec<(G::GameHash, Option<G::Move>)>,
        history_default: Vec<(G::GameHash, Option<G::Move>)>,
        winner: G::Player,
    ) {
        let z = if winner == self.color { 1. } else { 0. };
        let whole_history = [history.to_vec(), history_default].concat();
        for (t, (state, action)) in history.iter().enumerate() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;
            if let Some(action) = action {
                let mut v = node.moves.get_mut(action).unwrap();
                (*v).count += 1.;
                (*v).wins  += (z - (*v).wins) / (*v).count;

                // compute AMAF statistics
                for u in (t+2..whole_history.len()).step_by(2) {
                    if let (_, Some(action_u)) = whole_history[u] {
                        if (t..u)
                            .step_by(2)
                            .all(|i| Some(action_u) != whole_history[i].1)
                        {
                            if let Some(mut v_amaf) = node.moves.get_mut(&action_u) {
                                (*v_amaf).count_AMAF += 1.;
                                (*v_amaf).wins_AMAF  += (z - (*v_amaf).wins_AMAF) / (*v_amaf).count_AMAF;
                            }
                        }
                    }
                }
            }
        }
    }

    fn sim_tree(self: &mut RAVEPolicy<G>, b: &mut G) -> Vec<(G::GameHash, Option<G::Move>)> {
        let mut history: Vec<(G::GameHash, Option<G::Move>)> = Vec::new();

        while { b.winner() == None } {
            let s_t = b.hash();
            match self.tree.get(&s_t) {
                None => {
                    self.new_node(&b);
                    return history;
                }
                Some(_node) => {
                    if let Some(a) = self.select_move(&b) {
                        history.push((s_t, Some(a)));
                        b.play(&a)
                    } else {
                        history.push((s_t, None));
                        return history;
                    }
                }
            };
        }
        history
    }

    fn select_move(self: &RAVEPolicy<G>, board: &G) -> Option<G::Move> {
        let moves = board.possible_moves();

        if board.turn() == self.color {
            let mut max_move = None;
            let mut max_value = 0.;
            for _move in moves.iter() {
                let value = self.eval(&board.hash(), _move);
                if value >= max_value {
                    max_value = value;
                    max_move = Some(*_move);
                }
            };
            max_move
        } else {
            let mut min_move = None;
            let mut min_value = 1.;
            for _move in moves.iter() {
                let value = self.eval(&board.hash(), _move);
                if value <= min_value {
                    min_value = value;
                    min_move = Some(*_move);
                }
            }
            min_move
        }
    }

    fn eval(self: &RAVEPolicy<G>, state: &G::GameHash, action: &G::Move) -> f64{
        let node_info = self.tree.get(state).unwrap();
        let v = node_info.moves.get(action).unwrap();

        if v.count_AMAF == 0. {
            return v.wins;
        } else {
            let b = 0.4;
            let beta = v.count_AMAF / (v.count_AMAF + v.count + 4.*v.count_AMAF*v.count*b*b);
            return (1. - beta)*v.wins + beta * v.wins_AMAF;
        }
    }

    fn new_node(self: &mut RAVEPolicy<G>, board: &G) {
        let moves = HashMap::from_iter(board.possible_moves().into_iter().map(|m| {
            (
                m,
                MoveInfo {
                    wins: 1.,
                    wins_AMAF: 1.,
                    count: 0.,
                    count_AMAF: 0.,
                },
            )
        }));

        self.tree
            .insert(board.hash(), NodeInfo { count: 0., moves });
    }
}

impl<G: Game> Policy<G> for RAVEPolicy<G> {
    fn new(color: G::Player) -> RAVEPolicy<G> {
        RAVEPolicy {
            color,
            tree: HashMap::new(),
        }
    }

    fn play(self: &mut RAVEPolicy<G>, board: &G) -> Option<G::Move> {
        for _ in 0..N_PLAYOUTS {
            self.simulate(board)
        }

        let mut best_move = None;
        let mut max_visited = 0.;
        for m in board.possible_moves().iter() {
            let value = self.eval(&board.hash(), &m);

            let node_info = self.tree.get(&board.hash()).unwrap();
            let v = node_info.moves.get(&m).unwrap();

            println!("{:?} => {:?} ({:?})", value, m, v);
            if value >= max_visited {
                max_visited = value;
                best_move = Some(*m);
            }
        };
        best_move
    }
}
