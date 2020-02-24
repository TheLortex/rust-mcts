use rand::seq::SliceRandom;
use std::f64;
use std::iter::*;

use super::super::game::Game;
use super::{Policy, PolicyBuilder, N_PLAYOUTS};
use super::mcts::{UCTMoveInfo, UCTNodeInfo};

use std::collections::HashMap;
use std::marker::PhantomData;


pub struct PPAPolicy<G: Game, M: MoveCode<G>> {
    color: G::Player,
    s: PPA<G,M>,
    tree: HashMap<G::GameHash, UCTNodeInfo<G>>,
    playout_policy: HashMap<usize, f64>,

    _m: PhantomData<M>,
}

pub trait MoveCode<G: Game> {
    fn code(game: &G, action: &G::Move) -> usize;
} 

impl<G: Game, M: MoveCode<G>> PPAPolicy<G, M> {
    fn ensure_exists(self: &mut PPAPolicy<G, M>, board: &G) {
        for m in board.possible_moves() {
            let code = M::code(&board, &m);
            self.playout_policy.entry(code).or_insert(0.);
        }
    }

    fn simulate(self: &mut PPAPolicy<G, M>, root_board: &G) {
        let mut board = root_board.clone(); // COPY BOARD
        let history_uct = self.sim_tree(&mut board);

        let mut history_playout = vec![];

        while {board.winner() == None} {
            
            self.ensure_exists(&board);

            let moves = board.possible_moves();

            let chosen_move = moves
                .choose_weighted(&mut rand::thread_rng(), |item| {
                    self.playout_policy.get(&M::code(&board, item)).unwrap().exp()
                })
                .unwrap();

            board.play(chosen_move);
            history_playout.push(*chosen_move);
        }
        
        let z = board.winner().unwrap();
        self.update(&history_uct, z);
        self.adapt(root_board, &history_uct, &history_playout, z);
        
    }

    fn adapt(self: &mut PPAPolicy<G,M>, board: &G, history_uct: &[(G::GameHash, G::Move)], history_playout: &[G::Move], winner: G::Player) {
        let mut board = board.clone();
        for (_, action) in history_uct {
            if board.turn() == winner {
                self.policy_update(&board, &action);
            }
            board.play(&action);
        }

        for action in history_playout {
            if board.turn() == winner {
                self.policy_update(&board, &action);
            }
            board.play(&action);
        }
    }

    fn policy_update(self: &mut PPAPolicy<G,M>, board: &G, action: &G::Move) {
        let node = self.playout_policy.get_mut(&M::code(board, action)).unwrap();
        *node += self.s.alpha;

        let z: f64 = board
            .possible_moves()
            .iter()
            .map(|m| self.playout_policy.get(&M::code(board, &m)).unwrap().exp())
            .sum();
                
        for m in board.possible_moves() {
            let move_node = self.playout_policy.get_mut(&M::code(board, &m)).unwrap();
            let v = move_node.exp();
            *move_node -= self.s.alpha * v / z;
        }
    }

    fn update(
        self: &mut PPAPolicy<G, M>,
        history: &[(G::GameHash, G::Move)],
        winner: G::Player,
    ) {
        let z = if winner == self.color { 1. } else { 0. };
        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (z - (*v).Q) / (*v).N_a;
        }
    }

    fn sim_tree(self: &mut PPAPolicy<G, M>, b: &mut G) -> Vec<(G::GameHash, G::Move)> {
        let mut history: Vec<(G::GameHash, G::Move)> = Vec::new();

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
                        history.push((s_t, a));
                        b.play(&a)
                    } else { // surely there was an available move
                        panic!("? {} {:?}", b.possible_moves().len(), b);
                        //history.push((s_t, None));
                        //return history;
                    }
                }
            };
        }
        history
    }

    fn select_move(self: &PPAPolicy<G, M>, board: &G) -> Option<G::Move> {
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
                    v.Q + self.s.UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
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
                    v.Q - self.s.UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
                };

                if value <= min_value {
                    min_value = value;
                    min_move = Some(*_move);
                }
            }
            min_move
        }
    }

    pub fn new_node(self: &mut PPAPolicy<G, M>, board: &G) {
        let moves = HashMap::from_iter(
            board
                .possible_moves()
                .into_iter()
                .map(|m| (m, UCTMoveInfo { Q: 0., N_a: 0. })),
        );

        self.tree
            .insert(board.hash(), UCTNodeInfo { count: 0., moves });
    }
}

impl<G: Game, M: MoveCode<G>> Policy<G> for PPAPolicy<G, M> {
    fn play(self: &mut PPAPolicy<G, M>, board: &G) -> G::Move {
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


// POLICY BUILDER

pub struct PPA<G: Game, M:MoveCode<G>> {
    UCT_WEIGHT: f64,
    alpha: f64,
    _m: PhantomData<M>,
    _g: PhantomData<G>
}

impl<G: Game, M: MoveCode<G>> Copy for PPA<G, M> {}

impl<G: Game, M: MoveCode<G>> Clone for PPA<G, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Game, M: MoveCode<G>> Default for PPA<G,M> {
    fn default() -> PPA<G,M> {
        PPA::<G,M> {
            alpha: 0.1,
            UCT_WEIGHT: 0.4,
            _m: PhantomData,
            _g: PhantomData
        }
    }
}

impl<G: Game, M: MoveCode<G>> PolicyBuilder<G> for PPA<G,M> {
    type P = PPAPolicy<G, M>;

    fn create(&self, color: G::Player) -> PPAPolicy<G, M> {
        assert_eq!(G::players().len(), 2);
        PPAPolicy::<G,M> {
            color,
            s: *self,
            playout_policy: HashMap::new(),
            tree: HashMap::new(),
            _m: PhantomData
        }
    }
}
