use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::f64;
use std::iter::*;

use super::game::*;

const N_PLAYOUTS: usize = 100;
pub trait Policy {
    fn new(color: Color) -> Self;
    fn play(&mut self, board: &Board) -> Option<Move>;
}

pub struct RandomPolicy {
    color: Color,
}

impl Policy for RandomPolicy {
    fn new(color: Color) -> RandomPolicy {
        RandomPolicy { color }
    }

    fn play(self: &mut RandomPolicy, board: &Board) -> Option<Move> {
        let moves = board.possible_moves();
        moves.choose(&mut rand::thread_rng()).map(|x| *x)
    }
}

pub struct FlatMonteCarloPolicy {
    color: Color,
}

impl Policy for FlatMonteCarloPolicy {
    fn new(color: Color) -> FlatMonteCarloPolicy {
        FlatMonteCarloPolicy { color }
    }

    fn play(self: &mut FlatMonteCarloPolicy, board: &Board) -> Option<Move> {
        const FLAT_PLAYOUTS: usize = N_PLAYOUTS;

        let moves = board.possible_moves();

        if moves.len() > 0 {
            let n_playouts_per_move = FLAT_PLAYOUTS / moves.len();

            let mut best_move = None;
            let mut best_score = 0;

            for m in moves.iter() {
                let mut b_after_move = *board;
                b_after_move.play(&m);
                let mut success = 0;
                for _ in 0..n_playouts_per_move {
                    if b_after_move.playout(false) == self.color {
                        success += 1;
                    }
                }

                if success >= best_score {
                    best_score = success;
                    best_move = Some(m);
                }
            }

            best_move.map(|x| *x)
        } else {
            None
        }
    }
}

pub struct FlatUCBMonteCarloPolicy {
    color: Color,
}

impl Policy for FlatUCBMonteCarloPolicy {
    fn new(color: Color) -> FlatUCBMonteCarloPolicy {
        FlatUCBMonteCarloPolicy { color }
    }

    fn play(self: &mut FlatUCBMonteCarloPolicy, board: &Board) -> Option<Move> {
        const UCB_WEIGHT: f64 = 0.4;
        const UCB_PLAYOUTS: usize = N_PLAYOUTS;

        let moves = board.possible_moves();
        let mut move_success = HashMap::new();
        let mut move_count = HashMap::new();
        let mut move_board = HashMap::new();

        for m in moves.iter() {
            let mut b_after_move = *board;
            b_after_move.play(&m);
            if b_after_move.playout(false) == self.color {
                move_success.insert(m, 1);
            } else {
                move_success.insert(m, 0);
            }
            move_count.insert(m, 1);
            move_board.insert(m, b_after_move);
        }

        for i in 0..(UCB_PLAYOUTS - moves.len()) {
            let mut max_ucb = 0f64;
            let mut max_move = None;

            for m in moves.iter() {
                let count = *move_count.get(&m).unwrap() as f64;
                let succ = *move_success.get(&m).unwrap() as f64;
                let mean = succ / count;
                let ucb = mean + UCB_WEIGHT * (((moves.len() + i) as f64).ln() / count).sqrt();

                if ucb >= max_ucb {
                    max_move = Some(m);
                    max_ucb = ucb;
                }
            }

            if let Some(max_move) = max_move {
                *move_count.get_mut(&max_move).unwrap() += 1;
                if move_board.get(&max_move).unwrap().playout(false) == self.color {
                    *move_success.get_mut(&max_move).unwrap() += 1;
                }
            }
        }

        let mut max_count = 0;
        let mut max_move = None;

        for m in moves.iter() {
            let count = *move_count.get(&m).unwrap();

            if count >= max_count {
                max_move = Some(m);
                max_count = count;
            }
        }
        max_move.map(|x| *x)
    }
}

#[derive(Debug)]
struct MoveInfo {
    Q: f64,
    N_a: f64,
}

#[derive(Debug)]
struct NodeInfo {
    count: f64,
    moves: HashMap<Move, MoveInfo>,
}

pub struct UCTPolicy {
    color: Color,
    tree: HashMap<usize, NodeInfo>,
}

const UCT_WEIGHT: f64 = 0.4;

impl UCTPolicy {
    fn simulate(self: &mut UCTPolicy, board: &Board) {
        let mut b = *board; // COPY BOARD
        let history = self.sim_tree(&mut b);
        let z = b.playout(false);
        self.update(history, z);
    }

    pub fn debug(self: &UCTPolicy, board: &Board) {
        println!("{:?}", self.tree.get(&board.hash()));
    }

    fn update(self: &mut UCTPolicy, history: Vec<(usize, Option<Move>)>, winner: Color) {
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

    fn sim_tree(self: &mut UCTPolicy, b: &mut Board) -> Vec<(usize, Option<Move>)> {
        let mut history: Vec<(usize, Option<Move>)> = Vec::new();

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

    fn select_move(self: &UCTPolicy, board: &Board) -> Option<Move> {
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
                    v.Q + UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
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
                    v.Q - UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
                };

                if value <= min_value {
                    min_value = value;
                    min_move = Some(*_move);
                }
            }
            min_move
        }
    }

    fn new_node(self: &mut UCTPolicy, board: &Board) {
        let moves = HashMap::from_iter(
            board
                .possible_moves()
                .into_iter()
                .map(|m| (m, MoveInfo { Q: 0., N_a: 0. })),
        );

        self.tree
            .insert(board.hash(), NodeInfo { count: 0., moves });
    }

    fn show_tree(self: &UCTPolicy, board: &Board, rec: usize) {
        if let Some (entry) = self.tree.get(&board.hash()) {
            println!("Count: {}", entry.count);
            board.show();
            let moves = board.possible_moves();
    
            for m in moves.iter() {
                for _ in 0..rec {
                    print!("#")
                }
                println!("{:?}", entry.moves.get(&m).unwrap());
                let mut b = *board;
                b.play(&m);
                self.show_tree(&b, rec+1)
            };
        }
    }
}

impl Policy for UCTPolicy {
    fn new(color: Color) -> UCTPolicy {
        UCTPolicy {
            color,
            tree: HashMap::new(),
        }
    }

    fn play(self: &mut UCTPolicy, board: &Board) -> Option<Move> {
        for _ in 0..N_PLAYOUTS {
            self.simulate(board)
        }
        
        let info: &NodeInfo = self.tree.get(&board.hash()).unwrap();

        let mut best_move = None;
        let mut max_visited = 0.;
        for m in board.possible_moves().iter() {
            let x: &MoveInfo = info.moves.get(m).unwrap();
            if x.N_a >= max_visited {
                max_visited = x.N_a;
                best_move = Some (*m);
            }
        }
        best_move
    }
}
