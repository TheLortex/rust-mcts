use rand::seq::SliceRandom;
use std::f64;
use std::iter::*;

use super::super::game::Game;
use super::{Policy, PolicyBuilder, N_PLAYOUTS};

use std::collections::HashMap;

#[derive(Debug)]
struct MoveInfo {
    weight: f64,
}

#[derive(Debug)]
struct NodeInfo<G: Game> {
    moves: HashMap<G::Move, MoveInfo>,
}

pub struct NRPAPolicy<G: Game> {
    color: G::Player,
    s: NRPA,
    tree: HashMap<G::GameHash, NodeInfo<G>>,
}

impl<G: Game> NRPAPolicy<G> {
    /*
    1 NRPA(level,pol):
    2   IF level==0: // base rollout policy
    3     node = root(), ply = 0, seq = {}
    4     WHILE num_children(node)>0:
    5       CHOOSE seq[ply] = child i with probability proportional to exp(pol[code(node,i)])
    7       node = child(node,seq[ply])
    8       ply += 1
    9     RETURN (score(node),seq)
    10
    11   ELSE:   // for nesting levels>=1
    12     best_score = -infinity
    13     FOR N iterations:
    14       (result,new) = NRPA(level-1,pol)
    15       IF result>=best_score THEN:
    16          best_score = result
    17          seq = new
    18       pol = Adapt(pol,seq)
    19    RETURN (best_score,seq)
    */
    fn nested(self: &mut NRPAPolicy<G>, board: &G, level: usize) -> (f64, Vec<G::Move>) {

        if level == 0 {
            let mut board = board.clone();

            let mut history = vec![];
            while { !board.possible_moves().is_empty() } {
                self.ensure_exists(&board);
                let branch = self.tree.get(&board.hash());

                let moves = board.possible_moves();

                let chosen_move = moves
                    .choose_weighted(&mut rand::thread_rng(), |item| {
                        branch
                            .and_then(|branch| branch.moves.get(item).map(|node| node.weight.exp()))
                            .unwrap_or_else(||{ 
                                println!("Board: {:?}", board);
                                println!("Possible moves: {:?}", moves);
                                panic!("{:?} => {:?}", item, branch.unwrap().moves)
                            })
                    })
                    .unwrap_or_else(|_| panic!("{:?}", branch));
                board.play(chosen_move);
                history.push(*chosen_move);
            }
            (board.score(self.color), history)
        } else {
            let mut best_score = 0.;
            let mut best_hist = vec![];
            for _ in 0..self.s.N {
                let (result, history) = self.nested(board, level - 1);
                if result > best_score {
                    best_score = result;
                    best_hist = history;
                }
                self.adapt(board, &best_hist);
            }
            (best_score, best_hist)
        }
    }
    /*
    21 Adapt(pol,seq):   // a gradient ascent step towards seq
    22  node = root(), pol’ = pol
    23 FOR ply=0 TO length(seq)-1:
    24   pol’[code(node,seq[ply])] += Alpha
    25   z = SUM exp(pol[code(node,i)]) over node’s children i
    26   FOR children i of node:
    27    pol’[code(node,i)] -= Alpha*exp(pol[code(node,i)])/z
    28   node = child(node,seq[ply])
    29  RETURN pol’
    */
    fn adapt(self: &mut NRPAPolicy<G>, board: &G, history: &[G::Move]) {
        let mut board = board.clone();
        for action in history {
            self.ensure_exists(&board);
            let board_node = self.tree.get_mut(&board.hash()).unwrap();

            let move_node = board_node.moves.get_mut(action).unwrap();
            move_node.weight += self.s.alpha;

            let z: f64 = board
                .possible_moves()
                .iter()
                .map(|m| board_node.moves.get(&m).unwrap().weight.exp())
                .sum();

            for m in board.possible_moves() {
                let move_node = board_node.moves.get_mut(&m).unwrap();
                let v = move_node.weight.exp();
                move_node.weight -= self.s.alpha * v / z;
            }

            board.play(action);
        }
    }

    fn ensure_exists(self: &mut NRPAPolicy<G>, board: &G) {
        self.tree.entry(board.hash()).or_insert({
            let mut node = NodeInfo {
                moves: HashMap::new(),
            };
            for m in board.possible_moves() {
                node.moves.insert(m, MoveInfo { weight: 0. });
            };
            node
        });
    }
}

impl<G: Game> Policy<G> for NRPAPolicy<G> {
    fn play(self: &mut NRPAPolicy<G>, board: &G) -> G::Move {
        let (_, policy) = self.nested(board, self.s.level);
        policy[0]
    }
}

#[derive(Copy, Clone)]
pub struct NRPA {
    N: usize,
    level: usize,
    alpha: f64,
}

impl Default for NRPA {
    fn default() -> NRPA {
        NRPA {
            N: 10,
            alpha: 0.5,
            level: 3,
        }
    }
}

impl<G: Game> PolicyBuilder<G> for NRPA {
    type P = NRPAPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        assert_eq!(G::players().len(), 1);
        NRPAPolicy {
            color,
            s: *self,
            tree: HashMap::new(),
        }
    }
}
