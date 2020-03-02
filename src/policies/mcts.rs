use std::collections::HashMap;
use std::f64;
use std::iter::*;

use super::super::game::Game;
use super::{Policy, PolicyBuilder, N_PLAYOUTS};

/* UCT */

#[derive(Debug)]
pub struct UCTMoveInfo {
    pub Q: f64,
    pub N_a: f64,
}

#[derive(Debug)]
pub struct UCTNodeInfo<G: Game> {
    pub count: f64,
    pub moves: HashMap<G::Move, UCTMoveInfo>,
}

pub struct UCTPolicy<G: Game> {
    color: G::Player,
    tree: HashMap<G::GameHash, UCTNodeInfo<G>>,
    UCT_WEIGHT: f64,
}

impl<G: Game> UCTPolicy<G> {
    fn simulate(self: &mut UCTPolicy<G>, board: &G) {
        let mut b = board.clone(); // COPY BOARD
        let history = self.sim_tree(&mut b);
        let z = b.playout();
        self.update(history, z);
    }

    fn update(
        self: &mut UCTPolicy<G>,
        history: Vec<(G::GameHash, Option<G::Move>)>,
        winner: G::Player,
    ) {
        let z = if winner == self.color { 1. } else { 0. };
        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;
            if let Some(action) = action {
                let mut v = node.moves.get_mut(action).unwrap();
                (*v).N_a += 1.;
                (*v).Q += (z - (*v).Q) / (*v).N_a;
            }
        }
    }

    fn sim_tree(self: &mut UCTPolicy<G>, b: &mut G) -> Vec<(G::GameHash, Option<G::Move>)> {
        let mut history: Vec<(G::GameHash, Option<G::Move>)> = Vec::new();

        while { b.winner() == None } {
            let s_t = b.hash();
            match self.tree.get(&s_t) {
                None => {
                    //history.push((s_t, None));
                    self.new_node(&b);
                    return history;
                }
                Some(_node) => {
                    if let Some(a) = self.select_move(&b) {
                        history.push((s_t, Some(a)));
                        b.play(&a)
                    } else {
                        //panic!("? {} {:?}", b.possible_moves().len(), b);
                        history.push((s_t, None));
                        return history;
                    }
                }
            };
        }
        history
    }

    fn select_move(self: &UCTPolicy<G>, board: &G) -> Option<G::Move> {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board.hash()).unwrap();

        let N = node_info.count;
        if board.turn() == self.color {
            let mut max_move = None;
            let mut max_value = 0.;
            for _move in moves.iter() {
                let v = node_info.moves.get(_move).unwrap();
                let value = if v.N_a == 0. {
                    2.0
                } else {
                    v.Q + self.UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
                };
                if value >= max_value {
                    max_value = value;
                    max_move = Some(*_move);
                }
            }
            max_move
        } else {
            let mut min_move = None;
            let mut min_value = 1.;
            for _move in moves.iter() {
                let v = node_info.moves.get(_move).unwrap();
                let value = if v.N_a == 0. {
                    0.
                } else {
                    v.Q - self.UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
                };

                if value <= min_value {
                    min_value = value;
                    min_move = Some(*_move);
                }
            }
            min_move
        }
    }

    pub fn new_node(self: &mut UCTPolicy<G>, board: &G) {
        let moves = HashMap::from_iter(
            board
                .possible_moves()
                .into_iter()
                .map(|m| (*m, UCTMoveInfo { Q: 0., N_a: 0. })),
        );

        self.tree
            .insert(board.hash(), UCTNodeInfo { count: 0., moves });
    }
}

impl<G: Game> Policy<G> for UCTPolicy<G> {
    fn play(self: &mut UCTPolicy<G>, board: &G) -> G::Move {
        for _ in 0..N_PLAYOUTS {
            self.simulate(board)
        }

        let info: &UCTNodeInfo<G> = self.tree.get(&board.hash()).unwrap();

        let mut best_move = None;
        let mut max_visited = 0.;
        for m in board.possible_moves().iter() {
            let x: &UCTMoveInfo = info.moves.get(m).unwrap();
            if x.N_a >= max_visited {
                max_visited = x.N_a;
                best_move = Some(*m);
            }
        }
        best_move.unwrap()
    }
}

pub struct UCT {
    UCT_WEIGHT: f64,
}

impl Default for UCT {
    fn default() -> UCT {
        UCT { UCT_WEIGHT: 0.4 }
    }
}

impl<G: Game> PolicyBuilder<G> for UCT {
    type P = UCTPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        UCTPolicy {
            color,
            tree: HashMap::new(),
            UCT_WEIGHT: self.UCT_WEIGHT,
        }
    }
}

/* RAVE */

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
    UCT_WEIGHT: f64,
}

impl<G: Game> RAVEPolicy<G> {
    fn simulate(self: &mut RAVEPolicy<G>, board: &G) {
        let mut b = board.clone(); // COPY BOARD
        let history = self.sim_tree(&mut b);
        let (z, history_default) = b.playout_history();
        self.update(history, history_default, z);
    }

    fn update(
        self: &mut RAVEPolicy<G>,
        history: Vec<(G::GameHash, G::Move)>,
        history_default: Vec<(G::GameHash, G::Move)>,
        winner: G::Player,
    ) {
        let z = if winner == self.color { 1. } else { 0. };
        let whole_history = [history.to_vec(), history_default].concat();
        for (t, (state, action)) in history.iter().enumerate() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;

            let mut v = node.moves.get_mut(action).unwrap();
            (*v).count += 1.;
            (*v).wins += (z - (*v).wins) / (*v).count;

            // compute AMAF statistics
            for u in (t + 2..whole_history.len()).step_by(2) {
                let (_, action_u) = whole_history[u];
                if (t..u)
                    .step_by(2)
                    .all(|i| action_u != whole_history[i].1)
                {
                    if let Some(mut v_amaf) = node.moves.get_mut(&action_u) {
                        (*v_amaf).count_AMAF += 1.;
                        (*v_amaf).wins_AMAF += (z - (*v_amaf).wins_AMAF) / (*v_amaf).count_AMAF;
                    }
                }
            }
        }
    }

    fn sim_tree(self: &mut RAVEPolicy<G>, b: &mut G) -> Vec<(G::GameHash, G::Move)> {
        let mut history: Vec<(G::GameHash, G::Move)> = Vec::new();

        while { b.winner() == None } {
            let s_t = b.hash();
            match self.tree.get(&s_t) {
                None => {
                    self.new_node(&b);
                    return history;
                }
                Some(_node) => {
                    let a = self.select_move(&b);
                    history.push((s_t, a));
                    b.play(&a)
                }
            };
        }
        history
    }

    /*assumes there's at least one move to play */
    fn select_move(self: &RAVEPolicy<G>, board: &G) -> G::Move {
        let moves = board.possible_moves();

        if board.turn() == self.color {
            let mut max_move = None;
            let mut max_value = 0.;
            for _move in moves.iter() {
                let value = self.eval(&board.hash(), _move, true);
                if value >= max_value {
                    max_value = value;
                    max_move = Some(*_move);
                }
            }
            max_move.unwrap()
        } else {
            let mut min_move = None;
            let mut min_value = 1.;
            for _move in moves.iter() {
                let value = self.eval(&board.hash(), _move, false);
                if value <= min_value {
                    min_value = value;
                    min_move = Some(*_move);
                }
            }
            min_move.unwrap()
        }
    }

    fn beta(v: &MoveInfo) -> f64 {
        let b = 0.0001;
        let mut div = v.count_AMAF + v.count + 4. * v.count_AMAF * v.count * b * b;
        if div == 0. {
            div = 1.
        };
        v.count_AMAF / div
    }

    fn eval(self: &RAVEPolicy<G>, state: &G::GameHash, action: &G::Move, optimistic: bool) -> f64 {
        let node_info = self.tree.get(state).unwrap();
        let v = node_info.moves.get(action).unwrap();

        let multiplier = if optimistic { 1. } else { -1. };
        let v_mean =
            v.wins + multiplier * self.UCT_WEIGHT * (node_info.count.ln() / (1. + v.count)).sqrt();
        let v_AMAF = v.wins_AMAF;

        let beta = Self::beta(v);
        (1. - beta) * v_mean + beta * v_AMAF
    }

    pub fn new_node(self: &mut RAVEPolicy<G>, board: &G) {
        let moves = HashMap::from_iter(board.possible_moves().into_iter().map(|m| {
            (
                *m,
                MoveInfo {
                    wins: 0.,
                    wins_AMAF: 0.,
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
    fn play(self: &mut RAVEPolicy<G>, board: &G) -> G::Move {
        self.tree.clear();
        for _ in 0..N_PLAYOUTS {
            self.simulate(board)
        }

        let mut best_move = None;
        let mut max_visited = 0.;
        let mut _max_beta = 0.;
        for m in board.possible_moves().iter() {
            let value = self.eval(&board.hash(), &m, true);

            let node_info = self.tree.get(&board.hash()).unwrap();
            let v = node_info.moves.get(&m).unwrap();
            let beta = Self::beta(v);

            if value >= max_visited {
                max_visited = value;
                best_move = Some(*m);
                _max_beta = beta;
            }
        }
        best_move.unwrap()
    }
}

pub struct RAVE {
    UCT_WEIGHT: f64,
}

impl Default for RAVE {
    fn default() -> RAVE {
        RAVE { UCT_WEIGHT: 0.4 }
    }
}

impl<G: Game> PolicyBuilder<G> for RAVE {
    type P = RAVEPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        RAVEPolicy {
            color,
            tree: HashMap::new(),
            UCT_WEIGHT: self.UCT_WEIGHT,
        }
    }
}
