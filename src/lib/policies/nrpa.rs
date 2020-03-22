use super::super::game::{SingleplayerGame, MoveCode};
use super::{SingleplayerPolicy, SingleplayerPolicyBuilder};

use rand::seq::SliceRandom;
use std::f32;
use std::iter::*;
use std::marker::PhantomData;
use std::collections::HashMap;

pub struct NRPAPolicy<G: SingleplayerGame, M: MoveCode<G>> {
    s: NRPA<G,M>,
    _m: PhantomData<M>,
}

impl<G: SingleplayerGame, M: MoveCode<G>> NRPAPolicy<G, M> {

    fn next_move(playout_policy: &HashMap<usize, f32>, board: &G) -> G::Move {
        let moves = board.possible_moves().collect::<Vec<G::Move>>();
        let chosen_move = moves
            .choose_weighted(&mut rand::thread_rng(), |item| {
                let code = M::code(board, item);
                playout_policy.get(&code).unwrap_or(&0.).exp()
            })
            .unwrap();
        *chosen_move
    }


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
    //http://ieee-cog.org/2019/papers/paper_77.pdf
    fn nested(self: &mut NRPAPolicy<G, M>, board: &G, level: usize, mut playout_policy: HashMap<usize, f32>) -> (f32, Vec<G::Move>) {

        if level == 0 {
            let mut board = board.clone();

            let mut history = vec![];
            while { !board.is_finished() } {
                let chosen_move = Self::next_move(&playout_policy, &board);

                board.play(&chosen_move);
                history.push(chosen_move);
            }
            //println!("{:?}", history);
            (board.score(), history)
        } else {
            let mut best_score = 0.;
            let mut best_hist = vec![];


            for _ in 0..self.s.N {
                let (result, history) = self.nested(board, level - 1, playout_policy.clone());
                if result >= best_score {
                    best_score = result;
                    best_hist = history;
                }
                self.adapt(board, &best_hist, &mut playout_policy);
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
    fn adapt(self: &mut NRPAPolicy<G, M>, board: &G, history: &[G::Move], playout_policy: &mut HashMap<usize, f32>) {
        let mut board = board.clone();
        for action in history {
            let move_node = playout_policy.entry(M::code(&board, action)).or_insert(0.);
            *move_node += self.s.alpha;

            let z: f32 = board
                .possible_moves()
                .map(|m| playout_policy.get(&M::code(&board, &m)).unwrap_or(&0.).exp())
                .sum();

            for m in board.possible_moves() {
                let move_node = playout_policy.entry(M::code(&board, &m)).or_insert(0.);
                let v = move_node.exp();
                *move_node -= self.s.alpha * v / z;
            }

            board.play(action);
        }
    }
}

impl<G: SingleplayerGame, M: MoveCode<G>> SingleplayerPolicy<G> for NRPAPolicy<G, M> {
    fn solve(self: &mut NRPAPolicy<G, M>, board: &G) -> Vec<G::Move> {
        let (_, policy) = self.nested(board, self.s.level, HashMap::new());
        policy
    }
}

pub struct NRPA<G: SingleplayerGame, M: MoveCode<G>>  {
    N: usize,
    level: usize,
    alpha: f32,
    _m: PhantomData<M>,
    _g: PhantomData<G>
}

impl<G: SingleplayerGame, M: MoveCode<G>> Copy for NRPA<G, M> {}

impl<G: SingleplayerGame, M: MoveCode<G>> Clone for NRPA<G, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: SingleplayerGame, M: MoveCode<G>>  Default for NRPA<G,M> {
    fn default() -> NRPA<G,M> {
        NRPA::<G,M> {
            N: 1,
            alpha: 1.0,
            level: 3,
            _m: PhantomData,
            _g: PhantomData,
        }
    }
}

impl<G: SingleplayerGame, M: MoveCode<G>> SingleplayerPolicyBuilder<G> for NRPA<G,M> {
    type P = NRPAPolicy<G, M>;

    fn create(&self) -> Self::P {
        NRPAPolicy::<G,M> {
            s: *self,
            _m: PhantomData
        }
    }
}
